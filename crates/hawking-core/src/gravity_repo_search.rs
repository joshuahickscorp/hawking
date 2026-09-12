//! Bounded, deterministic repository text discovery for Gravity.
//!
//! This is the native replacement primitive for repeated Python `os.walk` +
//! whole-file search. It deliberately has no mutation authority and does not
//! follow symlinks. Callers choose explicit bounds; exceeding a bound is
//! surfaced as truncation rather than hidden work.

use crate::{Error, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, RwLock,
    },
    time::{Instant, UNIX_EPOCH},
};

const DEFAULT_MAX_FILE_BYTES: u64 = 2_000_000;
const DEFAULT_MAX_FILES: usize = 5_000;
const DEFAULT_MAX_RESULTS: usize = 100;
const DEFAULT_MAX_LIST_DIRECTORIES: usize = 10_000;

const SKIP_DIRS: &[&str] = &[
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".cache",
    "target",
    ".worktrees",
    "receipts",
    "workspace",
    "evidence",
    "build",
    "dist",
];
const SKIP_SUFFIXES: &[&str] = &[
    ".safetensors",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".onnx",
    ".npy",
    ".npz",
    ".so",
    ".dylib",
    ".a",
    ".o",
    ".zip",
    ".tar",
    ".gz",
    ".zst",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".mp4",
    ".mov",
    ".wav",
];
const CONTEXT_CODE_SUFFIXES: &[&str] = &[
    ".py", ".rs", ".ts", ".tsx", ".js", ".go", ".c", ".h", ".cpp", ".hpp", ".cu", ".metal",
    ".swift", ".sh", ".toml", ".md", ".json", ".yaml", ".yml",
];

/// Explicit resource budget for a read-only text search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoSearchLimits {
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub max_results: usize,
}

impl Default for RepoSearchLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_files: DEFAULT_MAX_FILES,
            max_results: DEFAULT_MAX_RESULTS,
        }
    }
}

/// One exact substring hit. `path` is relative to the supplied root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoSearchMatch {
    pub path: PathBuf,
    pub line: usize,
    pub text: String,
}

/// The complete bounded-search result; callers must respect `truncated`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoSearchResult {
    pub matches: Vec<RepoSearchMatch>,
    pub files_seen: usize,
    pub skipped_large: usize,
    pub truncated: bool,
}

/// One bounded filesystem-discovery row. Content is intentionally absent;
/// listing and reading are separate capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoListEntry {
    pub path: PathBuf,
    pub filename: String,
    pub entry_type: &'static str,
    pub size_bytes: Option<u64>,
}

/// Deterministic one-level or opt-in recursive filesystem discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoListResult {
    pub files: Vec<RepoListEntry>,
    pub directories: Vec<RepoListEntry>,
    pub truncated: bool,
    pub directories_seen: usize,
    pub elapsed_ns: u128,
}

/// Small ranked result for direct repository-path orientation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoPathRankResult {
    pub paths: Vec<PathBuf>,
    pub files_seen: usize,
    pub complete: bool,
    pub elapsed_ns: u128,
}

/// Small ranked result for content-oriented repository orientation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoContentRankResult {
    pub paths: Vec<PathBuf>,
    pub files_seen: usize,
    pub terms_used: usize,
    pub elapsed_ns: u128,
}

/// Bounded read result for a repository-contained text file.
///
/// Path validation remains an explicit caller boundary. This owner only
/// reads an already-contained regular file, bills the byte/window limits, and
/// returns the same provenance fields that the Python compatibility handler
/// exposes. It has no write or model authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RepoFileReadResult {
    pub path: PathBuf,
    pub bytes: u64,
    pub shown_bytes: usize,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation_note: Option<String>,
    pub sha256: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<usize>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub out_of_range: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RepoPathIndex {
    paths: Vec<PathBuf>,
    directory_stamps: Vec<FileStamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexedRepoFile {
    path: PathBuf,
    path_text: String,
    content: String,
    is_context_code: bool,
    is_test_path: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    path: PathBuf,
    size_bytes: u64,
    modified_ns: Option<u128>,
    is_directory: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RepoSearchFreshness {
    directories_fresh: bool,
    all_fresh: bool,
}

/// A bounded, immutable text index intended to live inside `hawkingd`.
///
/// Build cost is explicit. Queries avoid reopening or walking the repository,
/// which is the only credible path from cold hundreds-of-milliseconds scans to
/// millisecond-scale repeated discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoSearchIndex {
    root: PathBuf,
    files: Vec<IndexedRepoFile>,
    file_stamps: Vec<FileStamp>,
    directory_stamps: Vec<FileStamp>,
    /// Exact identifier-like token -> sorted file positions. This is the hot
    /// query lane; phrase/sub-string searches retain the bounded full scan.
    term_files: HashMap<String, Vec<usize>>,
    /// Three-byte token n-gram -> sorted file positions. This is a safe
    /// candidate prefilter for identifier-like substring searches; the final
    /// line scan still applies the requested case-sensitive substring match.
    term_trigram_files: HashMap<[u8; 3], Vec<usize>>,
    /// Same index restricted to context-code files. Content orientation never
    /// considers binary-adjacent/document-only files, so keeping that filter
    /// in the immutable index removes a per-hit classification branch from
    /// the resident query path.
    context_term_files: HashMap<String, Vec<usize>>,
    skipped_large: usize,
    truncated_at_build: bool,
}

/// Reusable query-local storage for the resident content scorer.
///
/// `RepoSearchIndex` is immutable and may be read concurrently, so the
/// scratch arena belongs to the service rather than the index. The resident
/// JSONL child is request-serial today; keeping this lock explicit preserves a
/// correct boundary if a future transport adds concurrent requests.
#[derive(Default)]
struct ContentRankScratch {
    scores: Vec<usize>,
    marks: Vec<u32>,
    touched: Vec<usize>,
    scoped_touched: Vec<usize>,
    matches: Vec<usize>,
    selected: Vec<(usize, usize)>,
    epoch: u32,
}

#[derive(Default)]
struct SearchCandidateCache {
    generation: u64,
    relative_scope: PathBuf,
    glob: String,
    candidates: Vec<usize>,
}

impl ContentRankScratch {
    fn prepare(&mut self, file_count: usize) {
        if self.scores.len() != file_count {
            self.scores.resize(file_count, 0);
            self.marks.resize(file_count, 0);
            self.marks.fill(0);
            self.epoch = 0;
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.touched.clear();
        self.scoped_touched.clear();
        self.matches.clear();
        self.selected.clear();
    }
}

/// Measured result of atomically rebuilding a resident repository index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoSearchRebuild {
    pub generation: u64,
    pub indexed_files: usize,
    pub elapsed_ns: u128,
}

/// Process-resident owner for indexed Gravity discovery.
///
/// `rebuild` constructs an entire fresh index before acquiring the write lock,
/// so concurrent queries continue to see the previous coherent generation. A
/// daemon can schedule this after file events without making a model/tool call
/// wait on directory traversal.
pub struct RepoSearchService {
    root: PathBuf,
    limits: RepoSearchLimits,
    index: RwLock<RepoSearchIndex>,
    rank_scratch: Mutex<ContentRankScratch>,
    candidate_cache: Mutex<Option<SearchCandidateCache>>,
    generation: AtomicU64,
    path_index: RwLock<RepoPathIndex>,
    path_generation: AtomicU64,
}

impl RepoSearchService {
    /// Open the first coherent generation. Daemons should call this during
    /// controlled startup, not on a request critical path.
    pub fn open(root: impl Into<PathBuf>, limits: RepoSearchLimits) -> Result<Self> {
        let requested_root = root.into();
        let root = fs::canonicalize(&requested_root).map_err(|error| {
            Error::Gravity(format!(
                "cannot resolve repository search root {}: {error}",
                requested_root.display()
            ))
        })?;
        let index = RepoSearchIndex::build(&root, limits)?;
        let path_index = RepoPathIndex::build(&root)?;
        Ok(Self {
            root,
            limits,
            index: RwLock::new(index),
            rank_scratch: Mutex::new(ContentRankScratch::default()),
            candidate_cache: Mutex::new(None),
            generation: AtomicU64::new(1),
            path_index: RwLock::new(path_index),
            path_generation: AtomicU64::new(1),
        })
    }

    /// Query the last complete index generation without filesystem traversal.
    pub fn search(&self, needle: &str, max_results: usize) -> Result<RepoSearchResult> {
        self.index
            .read()
            .map_err(|_| Error::Gravity("repository index read lock poisoned".into()))?
            .search(needle, max_results)
    }

    /// Search a contained scope using the filesystem-tool contract. The
    /// resident index supplies bytes and deterministic traversal; the query
    /// still bills the requested glob, case-sensitive matching, result cap, and per-file
    /// cap, so it does not widen Python's read authority.
    pub fn search_files(
        &self,
        requested: &Path,
        needle: &str,
        glob: &str,
        max_results: usize,
        max_per_file: Option<usize>,
    ) -> Result<RepoSearchResult> {
        if needle.is_empty() {
            return Err(Error::Gravity(
                "repository search needle must not be empty".into(),
            ));
        }
        if max_results == 0 {
            return Err(Error::Gravity(
                "repository search max_results must be non-zero".into(),
            ));
        }
        let scope = resolve_contained_path(&self.root, requested)?;
        let relative_scope = scope.strip_prefix(&self.root).unwrap_or(Path::new(""));
        let index = self
            .index
            .read()
            .map_err(|_| Error::Gravity("repository index read lock poisoned".into()))?;
        let generation = self.generation.load(Ordering::Acquire);
        let candidate_files = {
            let mut cache = self
                .candidate_cache
                .lock()
                .map_err(|_| Error::Gravity("repository candidate cache lock poisoned".into()))?;
            let cached = cache.as_ref().filter(|entry| {
                entry.generation == generation
                    && entry.relative_scope.as_path() == relative_scope
                    && entry.glob == glob
            });
            if let Some(entry) = cached {
                entry.candidates.clone()
            } else {
                let candidates = index.matching_file_indexes(relative_scope, glob);
                *cache = Some(SearchCandidateCache {
                    generation,
                    relative_scope: relative_scope.to_path_buf(),
                    glob: glob.to_owned(),
                    candidates: candidates.clone(),
                });
                candidates
            }
        };
        let freshness = index.freshness_for(relative_scope, &candidate_files);
        drop(index);
        if !freshness.all_fresh {
            // Rebuild outside the read lock, publish a complete generation,
            // then query it. A content-only edit does not churn the separate
            // path index; layout edits rebuild both owners. This keeps source
            // evidence current without making every file save pay for a
            // second full directory traversal.
            if freshness.directories_fresh {
                self.rebuild_search_index()?;
            } else {
                self.rebuild()?;
            }
            return self
                .index
                .read()
                .map_err(|_| Error::Gravity("repository index read lock poisoned".into()))?
                .search_files(relative_scope, needle, glob, max_results, max_per_file);
        }
        self.index
            .read()
            .map_err(|_| Error::Gravity("repository index read lock poisoned".into()))?
            .search_files_with_candidates(needle, max_results, max_per_file, &candidate_files)
    }

    /// Rebuild and atomically publish a new index generation.
    pub fn rebuild(&self) -> Result<RepoSearchRebuild> {
        let started = Instant::now();
        let fresh = RepoSearchIndex::build(&self.root, self.limits)?;
        let fresh_path_index = RepoPathIndex::build(&self.root)?;
        let indexed_files = fresh.indexed_files();
        let mut index = self
            .index
            .write()
            .map_err(|_| Error::Gravity("repository index write lock poisoned".into()))?;
        *index = fresh;
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        *self
            .candidate_cache
            .lock()
            .map_err(|_| Error::Gravity("repository candidate cache lock poisoned".into()))? = None;
        drop(index);
        *self
            .path_index
            .write()
            .map_err(|_| Error::Gravity("repository path index write lock poisoned".into()))? =
            fresh_path_index;
        self.path_generation.fetch_add(1, Ordering::AcqRel);
        Ok(RepoSearchRebuild {
            generation,
            indexed_files,
            elapsed_ns: started.elapsed().as_nanos(),
        })
    }

    /// Rebuild only the content index after metadata proves that repository
    /// layout is unchanged. The fresh immutable index is built before the
    /// write lock is acquired, so readers continue seeing one coherent
    /// generation throughout the replacement.
    fn rebuild_search_index(&self) -> Result<RepoSearchRebuild> {
        let started = Instant::now();
        let fresh = RepoSearchIndex::build(&self.root, self.limits)?;
        let indexed_files = fresh.indexed_files();
        let mut index = self
            .index
            .write()
            .map_err(|_| Error::Gravity("repository index write lock poisoned".into()))?;
        *index = fresh;
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        *self
            .candidate_cache
            .lock()
            .map_err(|_| Error::Gravity("repository candidate cache lock poisoned".into()))? = None;
        Ok(RepoSearchRebuild {
            generation,
            indexed_files,
            elapsed_ns: started.elapsed().as_nanos(),
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn path_generation(&self) -> u64 {
        self.path_generation.load(Ordering::Acquire)
    }

    /// Rank content-oriented evidence in one resident request. The scoring
    /// mirrors the Python compatibility contract: rare terms carry more
    /// weight, optional HCLI scoping keeps repository context local, and test
    /// questions receive a small filename tie-breaker.
    pub fn rank_content_paths(
        &self,
        terms: &[String],
        scope_prefix: Option<&str>,
        test_focus: bool,
        max_results: usize,
    ) -> Result<RepoContentRankResult> {
        if max_results == 0 {
            return Err(Error::Gravity(
                "repository content ranking max_results must be non-zero".into(),
            ));
        }
        let started = Instant::now();
        let mut scratch = self
            .rank_scratch
            .lock()
            .map_err(|_| Error::Gravity("repository rank scratch lock poisoned".into()))?;
        let result = self
            .index
            .read()
            .map_err(|_| Error::Gravity("repository index read lock poisoned".into()))?
            .rank_content_paths(terms, scope_prefix, test_focus, max_results, &mut scratch);
        let mut result = result;
        result.elapsed_ns = started.elapsed().as_nanos();
        Ok(result)
    }

    /// List a contained directory without Python's recursive `os.walk` path.
    /// The operation is deliberately uncoupled from the text index: listing
    /// newly-created files remains accurate while search stays generation-
    /// stable. The request is still bounded and never reads file contents.
    pub fn list(
        &self,
        requested: &Path,
        glob: &str,
        max_results: usize,
        recursive: bool,
    ) -> Result<RepoListResult> {
        list_repo_directory(&self.root, requested, glob, max_results, recursive)
    }

    /// Rank direct path-like hints without exporting the whole source path
    /// inventory over JSONL. A low-confidence result is explicitly incomplete
    /// so the caller can retain the existing content-search fallback.
    pub fn rank_code_paths(
        &self,
        suffixes: &[String],
        words: &[String],
        max_results: usize,
    ) -> Result<RepoPathRankResult> {
        if max_results == 0 {
            return Err(Error::Gravity(
                "repository path ranking max_results must be non-zero".into(),
            ));
        }
        if suffixes.is_empty() {
            return Err(Error::Gravity(
                "repository path ranking suffixes must not be empty".into(),
            ));
        }
        let started = Instant::now();
        // An explicitly named relative path is stronger evidence than a
        // repository-wide ranking query. Validate only that path and avoid
        // paying the directory-stamp walk when the answer is already in the
        // user's question. Non-exact queries retain full layout validation.
        if let Some(path) = direct_code_path(&self.root, suffixes, words) {
            let files_seen = self
                .path_index
                .read()
                .map_err(|_| Error::Gravity("repository path index read lock poisoned".into()))?
                .files_seen(suffixes);
            return Ok(RepoPathRankResult {
                paths: vec![path],
                files_seen,
                complete: true,
                elapsed_ns: started.elapsed().as_nanos(),
            });
        }
        let fresh = self
            .path_index
            .read()
            .map_err(|_| Error::Gravity("repository path index read lock poisoned".into()))?
            .is_fresh();
        if !fresh {
            let rebuilt = RepoPathIndex::build(&self.root)?;
            *self.path_index.write().map_err(|_| {
                Error::Gravity("repository path index write lock poisoned".into())
            })? = rebuilt;
            self.path_generation.fetch_add(1, Ordering::AcqRel);
        }
        let mut result = self
            .path_index
            .read()
            .map_err(|_| Error::Gravity("repository path index read lock poisoned".into()))?
            .rank(suffixes, words, max_results);
        result.elapsed_ns = started.elapsed().as_nanos();
        Ok(result)
    }
}

impl RepoPathIndex {
    fn build(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref()).map_err(|error| {
            Error::Gravity(format!(
                "cannot resolve repository path index root: {error}"
            ))
        })?;
        if !root.is_dir() {
            return Err(Error::Gravity(format!(
                "repository path index root is not a directory: {}",
                root.display()
            )));
        }
        let mut index = Self {
            paths: Vec::new(),
            directory_stamps: Vec::new(),
        };
        index_path_tree(&root, &root, &mut index)?;
        Ok(index)
    }

    fn is_fresh(&self) -> bool {
        let stamps = self.directory_stamps.iter().collect::<Vec<_>>();
        parallel_all_stamps_fresh(&stamps)
    }

    fn rank(
        &self,
        suffixes: &[String],
        words: &[String],
        max_results: usize,
    ) -> RepoPathRankResult {
        let mut scores = Vec::new();
        let mut files_seen = 0;
        for relative in &self.paths {
            if !has_requested_suffix(relative, suffixes) {
                continue;
            }
            files_seen += 1;
            let score = score_code_path(relative, words);
            if score > 0 {
                scores.push((relative.clone(), score));
            }
        }
        scores.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| {
                    left.0
                        .to_string_lossy()
                        .len()
                        .cmp(&right.0.to_string_lossy().len())
                })
                .then_with(|| left.0.cmp(&right.0))
        });
        let best_score = scores.first().map(|(_, score)| *score).unwrap_or_default();
        let complete = best_score >= 20;
        let paths = if complete {
            scores
                .into_iter()
                .filter(|(_, score)| *score * 2 >= best_score)
                .take(max_results)
                .map(|(path, _)| path)
                .collect()
        } else {
            Vec::new()
        };
        RepoPathRankResult {
            paths,
            files_seen,
            complete,
            elapsed_ns: 0,
        }
    }

    fn files_seen(&self, suffixes: &[String]) -> usize {
        self.paths
            .iter()
            .filter(|path| has_requested_suffix(path, suffixes))
            .count()
    }
}

impl RepoSearchIndex {
    /// Build an immutable index from the configured active repository scope.
    pub fn build(root: impl AsRef<Path>, limits: RepoSearchLimits) -> Result<Self> {
        if limits.max_files == 0 || limits.max_file_bytes == 0 {
            return Err(Error::Gravity(
                "repository index limits must be non-zero".into(),
            ));
        }
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(Error::Gravity(format!(
                "repository index root is not a directory: {}",
                root.display()
            )));
        }
        let mut index = Self {
            root: root.to_path_buf(),
            files: Vec::new(),
            file_stamps: Vec::new(),
            directory_stamps: Vec::new(),
            term_files: HashMap::new(),
            term_trigram_files: HashMap::new(),
            context_term_files: HashMap::new(),
            skipped_large: 0,
            truncated_at_build: false,
        };
        index_files(root, root, limits, &mut index)?;
        Ok(index)
    }

    /// Query the resident index without filesystem traversal.
    pub fn search(&self, needle: &str, max_results: usize) -> Result<RepoSearchResult> {
        if needle.is_empty() {
            return Err(Error::Gravity(
                "repository search needle must not be empty".into(),
            ));
        }
        if max_results == 0 {
            return Err(Error::Gravity(
                "repository search max_results must be non-zero".into(),
            ));
        }
        let mut result = RepoSearchResult {
            matches: Vec::new(),
            files_seen: self.files.len(),
            skipped_large: self.skipped_large,
            truncated: self.truncated_at_build,
        };
        let folded_needle = needle.to_ascii_lowercase();
        let candidate_files = if is_indexable_token(&folded_needle) {
            self.term_files
                .get(&folded_needle)
                .cloned()
                .unwrap_or_default()
        } else {
            (0..self.files.len()).collect()
        };
        for file_index in candidate_files {
            let file = &self.files[file_index];
            for (line_index, line) in file.content.lines().enumerate() {
                if !line.to_ascii_lowercase().contains(&folded_needle) {
                    continue;
                }
                result.matches.push(RepoSearchMatch {
                    path: file.path.clone(),
                    line: line_index + 1,
                    text: truncate_line(line, 1_000),
                });
                if result.matches.len() >= max_results {
                    result.truncated = true;
                    return Ok(result);
                }
            }
        }
        Ok(result)
    }

    fn search_files(
        &self,
        relative_scope: &Path,
        needle: &str,
        glob: &str,
        max_results: usize,
        max_per_file: Option<usize>,
    ) -> Result<RepoSearchResult> {
        let candidate_files = self.matching_file_indexes(relative_scope, glob);
        self.search_files_with_candidates(needle, max_results, max_per_file, &candidate_files)
    }

    fn search_files_with_candidates(
        &self,
        needle: &str,
        max_results: usize,
        max_per_file: Option<usize>,
        candidate_files: &[usize],
    ) -> Result<RepoSearchResult> {
        let folded_needle = needle.to_ascii_lowercase();
        // The term index is a safe superset for exact Python matching: it is
        // built case-insensitively, then each line below is checked with the
        // requested case-sensitive substring semantics.
        let candidates = if is_indexable_token(&folded_needle) && folded_needle.len() >= 3 {
            self.substring_candidate_files(&folded_needle, candidate_files)
        } else {
            candidate_files.to_vec()
        };
        let mut result = RepoSearchResult {
            matches: Vec::new(),
            files_seen: candidate_files.len(),
            skipped_large: self.skipped_large,
            truncated: self.truncated_at_build,
        };
        for file_index in candidates {
            let file = &self.files[file_index];
            let mut file_hits = 0;
            for (line_index, line) in file.content.lines().enumerate() {
                if !line.contains(needle) {
                    continue;
                }
                result.matches.push(RepoSearchMatch {
                    path: file.path.clone(),
                    line: line_index + 1,
                    text: truncate_line(line, 1_000),
                });
                file_hits += 1;
                if result.matches.len() >= max_results {
                    result.truncated = true;
                    return Ok(result);
                }
                if max_per_file.is_some_and(|limit| file_hits >= limit) {
                    break;
                }
            }
        }
        Ok(result)
    }

    fn substring_candidate_files(
        &self,
        folded_needle: &str,
        candidate_files: &[usize],
    ) -> Vec<usize> {
        let grams = folded_needle
            .as_bytes()
            .windows(3)
            .map(|bytes| [bytes[0], bytes[1], bytes[2]])
            .collect::<BTreeSet<_>>();
        let Some(first) = grams
            .iter()
            .filter_map(|gram| self.term_trigram_files.get(gram))
            .min_by_key(|posting| posting.len())
        else {
            return Vec::new();
        };
        let mut candidates = first.clone();
        for gram in grams {
            let Some(posting) = self.term_trigram_files.get(&gram) else {
                return Vec::new();
            };
            if std::ptr::eq(posting, first) {
                continue;
            }
            candidates.retain(|index| posting.binary_search(index).is_ok());
            if candidates.is_empty() {
                return candidates;
            }
        }
        candidates.retain(|index| candidate_files.binary_search(index).is_ok());
        candidates
    }

    pub fn indexed_files(&self) -> usize {
        self.files.len()
    }

    fn matching_file_indexes(&self, relative_scope: &Path, glob: &str) -> Vec<usize> {
        (0..self.files.len())
            .filter(|index| {
                let path = &self.files[*index].path;
                (relative_scope.as_os_str().is_empty() || path.starts_with(relative_scope))
                    && path
                        .file_name()
                        .is_some_and(|name| simple_glob_match(&name.to_string_lossy(), glob))
            })
            .collect()
    }

    /// Check only metadata that can affect this scoped search. Directory
    /// stamps cover additions, removals, and renames; matching file stamps
    /// cover content changes that do not touch a directory mtime. Unrelated
    /// files are deliberately outside this query's evidence boundary, which
    /// keeps current-source validation from becoming an O(repository) tax on
    /// every narrow search.
    fn freshness_for(
        &self,
        relative_scope: &Path,
        candidate_files: &[usize],
    ) -> RepoSearchFreshness {
        let directory_stamps = self
            .directory_stamps
            .iter()
            .filter(|stamp| {
                let relative = stamp.path.strip_prefix(&self.root).unwrap_or(&stamp.path);
                directory_affects_scope(relative, relative_scope)
            })
            .collect::<Vec<_>>();
        let directories_fresh = parallel_all_stamps_fresh(&directory_stamps);
        if !directories_fresh {
            return RepoSearchFreshness {
                directories_fresh: false,
                all_fresh: false,
            };
        }
        let mut file_stamps = Vec::with_capacity(candidate_files.len());
        for index in candidate_files {
            let Some(stamp) = self.file_stamps.get(*index) else {
                return RepoSearchFreshness {
                    directories_fresh: true,
                    all_fresh: false,
                };
            };
            file_stamps.push(stamp);
        }
        RepoSearchFreshness {
            directories_fresh: true,
            all_fresh: parallel_all_stamps_fresh(&file_stamps),
        }
    }

    fn rank_content_paths(
        &self,
        terms: &[String],
        scope_prefix: Option<&str>,
        test_focus: bool,
        max_results: usize,
        scratch: &mut ContentRankScratch,
    ) -> RepoContentRankResult {
        // Keep stable file indexes in the scorer. Cloning and comparing a
        // PathBuf for every term made the ranking allocator-bound even though
        // the resident index already owns every path. Materialize paths only
        // for the bounded response at the end.
        // The resident index assigns every file a stable dense integer. A
        // BTreeMap made each query pay ordered-map allocation and log-N
        // updates even though the final response is sorted separately. Keep
        // only touched indexes, with a dense score vector for accumulation;
        // the final sort below remains the deterministic ordering authority.
        scratch.prepare(self.files.len());
        let mut terms_used = 0;
        for term in terms {
            let folded_term = term.to_ascii_lowercase();
            if is_indexable_token(&folded_term) {
                let Some(candidates) = self.context_term_files.get(&folded_term) else {
                    continue;
                };
                let candidates = &candidates[..candidates.len().min(1_000)];
                if candidates.is_empty() {
                    continue;
                }
                terms_used += 1;
                let weight = match candidates.len() {
                    0..=10 => 8,
                    11..=50 => 4,
                    51..=200 => 2,
                    _ => 1,
                };
                self.accumulate_rank_candidates(candidates, weight, scope_prefix, scratch);
                continue;
            }
            self.content_file_indexes_into(term, 1_000, &mut scratch.matches);
            if scratch.matches.is_empty() {
                continue;
            }
            terms_used += 1;
            let weight = match scratch.matches.len() {
                0..=10 => 8,
                11..=50 => 4,
                51..=200 => 2,
                _ => 1,
            };
            let epoch = scratch.epoch;
            for &file_index in &scratch.matches {
                if scratch.marks[file_index] != epoch {
                    scratch.marks[file_index] = epoch;
                    scratch.scores[file_index] = 0;
                    scratch.touched.push(file_index);
                    if let Some(prefix) = scope_prefix {
                        if self
                            .files
                            .get(file_index)
                            .is_some_and(|file| file.path_text.starts_with(prefix))
                        {
                            scratch.scoped_touched.push(file_index);
                        }
                    }
                }
                scratch.scores[file_index] += weight;
            }
        }
        let files_seen = scratch.touched.len();
        // A scoped query only needs to rank the scoped hits when any exist.
        // The old path materialized and sorted every hit, then discarded the
        // non-scoped rows. Keep the fallback semantics (no scoped hit means
        // rank the complete hit set), but avoid that work when the scope is
        // populated.
        let selected_files = if scope_prefix.is_some() && !scratch.scoped_touched.is_empty() {
            &scratch.scoped_touched
        } else {
            &scratch.touched
        };
        scratch.selected.extend(
            selected_files
                .iter()
                .map(|&file_index| (file_index, scratch.scores[file_index])),
        );
        if test_focus {
            for (file_index, score) in &mut scratch.selected {
                if self
                    .files
                    .get(*file_index)
                    .is_some_and(|file| file.is_test_path)
                {
                    *score += 3;
                }
            }
        }
        let mut compare = |left: &(usize, usize), right: &(usize, usize)| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| {
                    self.files[left.0]
                        .path_text
                        .len()
                        .cmp(&self.files[right.0].path_text.len())
                })
                .then_with(|| {
                    self.files[left.0]
                        .path_text
                        .cmp(&self.files[right.0].path_text)
                })
        };
        // HCLI normally asks for four paths. Selecting the bounded prefix
        // avoids sorting every scoped hit while preserving the exact total
        // ordering for the returned rows. Keep the full sort when the caller
        // requests at least the complete candidate set.
        let keep = max_results.min(scratch.selected.len());
        if keep > 0 && keep < scratch.selected.len() {
            scratch
                .selected
                .select_nth_unstable_by(keep - 1, &mut compare);
            scratch.selected.truncate(keep);
        }
        scratch.selected.sort_unstable_by(&mut compare);
        RepoContentRankResult {
            paths: scratch
                .selected
                .iter()
                .map(|&(file_index, _)| self.files[file_index].path.clone())
                .collect(),
            files_seen,
            terms_used,
            elapsed_ns: 0,
        }
    }

    fn accumulate_rank_candidates(
        &self,
        candidates: &[usize],
        weight: usize,
        scope_prefix: Option<&str>,
        scratch: &mut ContentRankScratch,
    ) {
        let epoch = scratch.epoch;
        for &file_index in candidates {
            if scratch.marks[file_index] != epoch {
                scratch.marks[file_index] = epoch;
                scratch.scores[file_index] = 0;
                scratch.touched.push(file_index);
                if let Some(prefix) = scope_prefix {
                    if self
                        .files
                        .get(file_index)
                        .is_some_and(|file| file.path_text.starts_with(prefix))
                    {
                        scratch.scoped_touched.push(file_index);
                    }
                }
            }
            scratch.scores[file_index] += weight;
        }
    }

    fn content_file_indexes_into(&self, term: &str, max_matches: usize, indexes: &mut Vec<usize>) {
        indexes.clear();
        let folded_term = term.to_ascii_lowercase();
        if is_indexable_token(&folded_term) {
            // `indexable_terms` records exact identifier-like tokens from the
            // same lowercased content used by `search`. For this class, the
            // term index is already a proof of a case-insensitive substring
            // hit; re-reading every matching line only duplicates work.
            if let Some(candidates) = self.context_term_files.get(&folded_term) {
                for &index in candidates {
                    indexes.push(index);
                    if indexes.len() >= max_matches {
                        break;
                    }
                }
            }
            return;
        }
        let mut line_matches = 0;
        'files: for file_index in 0..self.files.len() {
            let Some(file) = self.files.get(file_index) else {
                continue;
            };
            if !file.is_context_code {
                continue;
            }
            let mut file_matches = false;
            for line in file.content.lines() {
                if !line.to_ascii_lowercase().contains(&folded_term) {
                    continue;
                }
                file_matches = true;
                line_matches += 1;
                if line_matches >= max_matches {
                    break;
                }
            }
            if file_matches {
                indexes.push(file_index);
            }
            if line_matches >= max_matches {
                break 'files;
            }
        }
    }
}

fn directory_affects_scope(relative_directory: &Path, relative_scope: &Path) -> bool {
    relative_scope.as_os_str().is_empty()
        || relative_directory.starts_with(relative_scope)
        || relative_scope.starts_with(relative_directory)
}

fn stamp_matches_metadata(stamp: &FileStamp) -> bool {
    let Ok(metadata) = fs::symlink_metadata(&stamp.path) else {
        return false;
    };
    metadata.file_type().is_dir() == stamp.is_directory
        && metadata.len() == stamp.size_bytes
        && modified_ns(&metadata) == stamp.modified_ns
}

/// Validate a large stamp set without making one request wait on a serial
/// metadata syscall chain. The work is read-only and commutative: every stamp
/// must match, so bounded scoped workers cannot change the result or ordering.
/// Small scopes stay single-threaded to avoid paying worker setup overhead.
fn parallel_all_stamps_fresh(stamps: &[&FileStamp]) -> bool {
    // The contained HCLI search scope commonly carries a few hundred source
    // files.  On the local SSD, checking that set with bounded parallelism is
    // faster than a serial metadata chain while retaining the exact
    // all-stamps-must-match result.  Keep genuinely small scopes serial to
    // avoid paying worker setup overhead.
    const SEQUENTIAL_LIMIT: usize = 128;
    const STAMPS_PER_WORKER: usize = 128;
    const MAX_WORKERS: usize = 8;
    if stamps.len() <= SEQUENTIAL_LIMIT {
        return stamps.iter().all(|stamp| stamp_matches_metadata(stamp));
    }
    let workers = stamps.len().div_ceil(STAMPS_PER_WORKER).min(MAX_WORKERS);
    let chunk_size = stamps.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles = stamps
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || chunk.iter().all(|stamp| stamp_matches_metadata(stamp)))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .all(|handle| handle.join().unwrap_or(false))
    })
}

/// Search regular text files in deterministic lexical traversal order.
pub fn search_repo(
    root: impl AsRef<Path>,
    needle: &str,
    limits: RepoSearchLimits,
) -> Result<RepoSearchResult> {
    if needle.is_empty() {
        return Err(Error::Gravity(
            "repository search needle must not be empty".into(),
        ));
    }
    if limits.max_files == 0 || limits.max_results == 0 || limits.max_file_bytes == 0 {
        return Err(Error::Gravity(
            "repository search limits must be non-zero".into(),
        ));
    }
    let root = root.as_ref();
    if !root.is_dir() {
        return Err(Error::Gravity(format!(
            "repository search root is not a directory: {}",
            root.display()
        )));
    }
    RepoSearchIndex::build(root, limits)?.search(needle, limits.max_results)
}

fn resolve_contained_path(root: &Path, requested: &Path) -> Result<PathBuf> {
    let resolved = fs::canonicalize(requested).map_err(|error| {
        Error::Gravity(format!(
            "cannot resolve repository-scoped path {}: {error}",
            requested.display()
        ))
    })?;
    if !resolved.starts_with(root) {
        return Err(Error::Gravity(format!(
            "repository path is outside the native root: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

/// Read one existing, repository-contained regular file with the bounded
/// semantics used by HCLI's fs.read operation.
pub fn read_repo_file(
    root: impl AsRef<Path>,
    requested: impl AsRef<Path>,
    max_bytes: usize,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<RepoFileReadResult> {
    const MAX_READ_BYTES: usize = 2 * 1024 * 1024;
    const RESULT_STRING_LIMIT: usize = 4_000;
    if max_bytes == 0 {
        return Err(Error::Gravity(
            "repository read max_bytes must be non-zero".into(),
        ));
    }
    let root = fs::canonicalize(root.as_ref()).map_err(|error| {
        Error::Gravity(format!(
            "cannot resolve repository read root {}: {error}",
            root.as_ref().display()
        ))
    })?;
    let requested = requested.as_ref();
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let path = resolve_contained_path(&root, &requested)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err(Error::Gravity(format!(
            "repository read path is not a regular file: {}",
            path.display()
        )));
    }

    let raw = fs::read(&path)?;
    let bytes = raw.len() as u64;
    let sha256 = sha256_hex(&raw);
    let limit = max_bytes.min(MAX_READ_BYTES).max(1);
    let (content_bytes, start_line, end_line, total_lines, out_of_range, window_note) =
        if start_line.is_some() || end_line.is_some() {
            let text = String::from_utf8_lossy(&raw);
            let lines = split_lines_keepends(&text);
            let first = start_line.unwrap_or(1).max(1);
            let last = end_line.unwrap_or(lines.len()).min(lines.len());
            let out_of_range = first > lines.len();
            let selected = if first <= last {
                lines[first - 1..last].concat()
            } else {
                String::new()
            };
            let note = if out_of_range {
                Some(format!(
                    "EMPTY WINDOW: this file has {} lines, so the window you asked for \
(starting at line {}) does not exist. The empty content below is NOT the file's \
content -- there is nothing there. Read within 1..{}.",
                    lines.len(),
                    first,
                    lines.len()
                ))
            } else {
                None
            };
            (
                selected.into_bytes(),
                Some(first),
                Some(if out_of_range { first } else { last }),
                Some(lines.len()),
                out_of_range,
                note,
            )
        } else {
            (raw.clone(), None, None, None, false, None)
        };
    let clipped = &content_bytes[..content_bytes.len().min(limit)];
    let shown_bytes = clipped.len().min(RESULT_STRING_LIMIT);
    let truncation_note = if shown_bytes >= content_bytes.len() {
        None
    } else {
        let pct = if content_bytes.is_empty() {
            0.0
        } else {
            shown_bytes as f64 / content_bytes.len() as f64 * 100.0
        };
        Some(format!(
            "TRUNCATED: you are seeing {} of {} bytes ({:.1}%). Do not conclude \
anything about what is NOT shown. Narrow with start/end lines, or use the \
purpose-built tool for this file if one exists.",
            shown_bytes,
            content_bytes.len(),
            pct
        ))
    };
    Ok(RepoFileReadResult {
        path,
        bytes,
        shown_bytes,
        truncated: truncation_note.is_some(),
        truncation_note,
        sha256,
        content: String::from_utf8_lossy(clipped).into_owned(),
        start_line,
        end_line,
        total_lines,
        out_of_range,
        window_note,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

/// Python splitlines(keepends=True) behavior for the line endings found in
/// source files. Newline bytes are ASCII, so slicing remains on UTF-8 bounds.
fn split_lines_keepends(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let end = match bytes[index] {
            b'\n' => Some(index + 1),
            b'\r' => Some(if bytes.get(index + 1) == Some(&b'\n') {
                index + 2
            } else {
                index + 1
            }),
            _ => None,
        };
        if let Some(end) = end {
            lines.push(&text[start..end]);
            start = end;
            index = end;
        } else {
            index += 1;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Bounded filesystem discovery used by the resident Gravity service.
pub fn list_repo_directory(
    root: impl AsRef<Path>,
    requested: impl AsRef<Path>,
    glob: &str,
    max_results: usize,
    recursive: bool,
) -> Result<RepoListResult> {
    if max_results == 0 {
        return Err(Error::Gravity(
            "repository list max_results must be non-zero".into(),
        ));
    }
    let root = fs::canonicalize(root.as_ref())
        .map_err(|error| Error::Gravity(format!("cannot resolve repository list root: {error}")))?;
    let requested = requested.as_ref();
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let directory = resolve_contained_path(&root, &requested)?;
    if !directory.is_dir() {
        return Err(Error::Gravity(format!(
            "repository list path is not a directory: {}",
            directory.display()
        )));
    }

    let started = Instant::now();
    let mut result = RepoListResult {
        files: Vec::new(),
        directories: Vec::new(),
        truncated: false,
        directories_seen: 0,
        elapsed_ns: 0,
    };
    list_directory_inner(
        &directory,
        &directory,
        glob,
        max_results.min(2_000),
        recursive,
        &mut result,
    )?;
    result.elapsed_ns = started.elapsed().as_nanos();
    Ok(result)
}

fn index_path_tree(root: &Path, directory: &Path, index: &mut RepoPathIndex) -> Result<()> {
    let metadata = fs::metadata(directory).map_err(|error| {
        Error::Gravity(format!("cannot inspect {}: {error}", directory.display()))
    })?;
    index.directory_stamps.push(FileStamp {
        path: directory.to_path_buf(),
        size_bytes: metadata.len(),
        modified_ns: modified_ns(&metadata),
        is_directory: true,
    });

    let mut entries = fs::read_dir(directory)
        .map_err(|error| Error::Gravity(format!("cannot list {}: {error}", directory.display())))?
        .filter_map(|entry| entry.ok())
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let hidden = path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with('.'));
            if hidden
                || path
                    .file_name()
                    .is_some_and(|name| SKIP_DIRS.iter().any(|skip| name == *skip))
            {
                continue;
            }
            index_path_tree(root, &path, index)?;
            continue;
        }
        if !file_type.is_file() || has_skipped_suffix(&path) {
            continue;
        }
        index
            .paths
            .push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
    }
    Ok(())
}

fn has_requested_suffix(path: &Path, suffixes: &[String]) -> bool {
    let Some(extension) = path.extension() else {
        return false;
    };
    let extension = extension.to_string_lossy();
    suffixes.iter().any(|suffix| {
        suffix
            .strip_prefix('.')
            .is_some_and(|expected| expected == extension)
    })
}

fn direct_code_path(root: &Path, suffixes: &[String], words: &[String]) -> Option<PathBuf> {
    words.iter().find_map(|word| {
        if !word.contains('/') {
            return None;
        }
        let candidate = Path::new(word);
        if candidate.is_absolute()
            || candidate
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            || !has_requested_suffix(candidate, suffixes)
            || candidate.parent().is_some_and(path_has_skipped_directory)
        {
            return None;
        }
        let absolute = root.join(candidate);
        let metadata = fs::symlink_metadata(&absolute).ok()?;
        if !metadata.file_type().is_file() {
            return None;
        }
        let resolved = fs::canonicalize(&absolute).ok()?;
        let relative = resolved.strip_prefix(root).ok()?;
        if relative != candidate {
            return None;
        }
        Some(candidate.to_path_buf())
    })
}

fn path_has_skipped_directory(path: &Path) -> bool {
    path.components().any(|component| {
        let std::path::Component::Normal(name) = component else {
            return false;
        };
        let name = name.to_string_lossy();
        name.starts_with('.') || SKIP_DIRS.iter().any(|skip| *skip == name)
    })
}

fn score_code_path(relative: &Path, words: &[String]) -> usize {
    let lowered = relative.to_string_lossy().to_ascii_lowercase();
    let name = relative
        .file_name()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let stem = relative
        .file_stem()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    words.iter().fold(0, |score, word| {
        let word = word.to_ascii_lowercase();
        if word == lowered {
            score + 25
        } else if lowered.ends_with(&format!("/{word}")) || word == name {
            score + 20
        } else if word == stem {
            score + 10
        } else if lowered.contains(&word) {
            score + 3
        } else {
            score
        }
    })
}

fn list_directory_inner(
    directory: &Path,
    base: &Path,
    glob: &str,
    max_results: usize,
    recursive: bool,
    result: &mut RepoListResult,
) -> Result<()> {
    if result.directories_seen >= DEFAULT_MAX_LIST_DIRECTORIES {
        result.truncated = true;
        return Ok(());
    }
    result.directories_seen += 1;

    let mut rows = match fs::read_dir(directory) {
        Ok(rows) => rows
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let metadata = fs::symlink_metadata(&path).ok()?;
                Some((
                    path.file_name()?.to_string_lossy().into_owned(),
                    path,
                    metadata,
                ))
            })
            .collect::<Vec<_>>(),
        Err(_) => return Ok(()),
    };
    rows.sort_by(|left, right| left.0.cmp(&right.0));

    let mut child_directories = Vec::new();
    for (filename, path, metadata) in rows {
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            if matches!(
                filename.as_str(),
                ".git" | ".venv" | "__pycache__" | "node_modules"
            ) {
                continue;
            }
            child_directories.push((filename, path));
            continue;
        }
        if !file_type.is_file() || !simple_glob_match(&filename, glob) {
            continue;
        }
        if result.files.len() >= max_results {
            result.truncated = true;
            continue;
        }
        let size_bytes = fs::metadata(&path).ok().map(|item| item.len());
        result.files.push(RepoListEntry {
            path: path.strip_prefix(base).unwrap_or(&path).to_path_buf(),
            filename,
            entry_type: "file",
            size_bytes,
        });
    }

    // Match the Python contract: directory rows and file rows have independent
    // caps, and every returned row has a deterministic lexical order.
    for (filename, path) in &child_directories {
        if !simple_glob_match(filename, glob) {
            continue;
        }
        if result.directories.len() >= max_results {
            result.truncated = true;
            continue;
        }
        result.directories.push(RepoListEntry {
            path: path.strip_prefix(base).unwrap_or(path).to_path_buf(),
            filename: filename.clone(),
            entry_type: "directory",
            size_bytes: None,
        });
    }

    if !recursive {
        return Ok(());
    }
    for (_filename, path) in child_directories {
        if result.files.len() >= max_results && result.directories.len() >= max_results {
            result.truncated = true;
            break;
        }
        list_directory_inner(&path, base, glob, max_results, recursive, result)?;
        if result.directories_seen >= DEFAULT_MAX_LIST_DIRECTORIES {
            break;
        }
    }
    Ok(())
}

/// Small deterministic glob matcher for the basename patterns exposed by
/// `fs.list` (`*`, `?`, and literal characters). Path separators are not
/// treated specially because the Python caller matches each basename.
fn simple_glob_match(value: &str, pattern: &str) -> bool {
    let value = value.as_bytes();
    let pattern = pattern.as_bytes();
    let mut value_index = 0;
    let mut pattern_index = 0;
    let mut star = None;
    let mut star_value = 0;
    while value_index < value.len() {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == b'?' || pattern[pattern_index] == value[value_index])
        {
            value_index += 1;
            pattern_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star = Some(pattern_index);
            pattern_index += 1;
            star_value = value_index;
        } else if let Some(star_index) = star {
            pattern_index = star_index + 1;
            star_value += 1;
            value_index = star_value;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

fn index_files(
    root: &Path,
    directory: &Path,
    limits: RepoSearchLimits,
    index: &mut RepoSearchIndex,
) -> Result<()> {
    let directory_metadata = fs::metadata(directory).map_err(|error| {
        Error::Gravity(format!("cannot inspect {}: {error}", directory.display()))
    })?;
    index.directory_stamps.push(FileStamp {
        path: directory.to_path_buf(),
        size_bytes: directory_metadata.len(),
        modified_ns: modified_ns(&directory_metadata),
        is_directory: true,
    });
    let mut entries = BTreeSet::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| Error::Gravity(format!("cannot list {}: {error}", directory.display())))?
    {
        let entry = entry.map_err(|error| {
            Error::Gravity(format!("cannot inspect {}: {error}", directory.display()))
        })?;
        entries.insert(entry.path());
    }
    for path in entries {
        if index.truncated_at_build {
            return Ok(());
        }
        let file_type = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata.file_type(),
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if path.file_name().is_some_and(|name| {
                name.to_string_lossy().starts_with('.')
                    || SKIP_DIRS.iter().any(|skip| name == *skip)
            }) {
                continue;
            }
            index_files(root, &path, limits, index)?;
        } else if file_type.is_file() && !has_skipped_suffix(&path) {
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            if metadata.len() > limits.max_file_bytes {
                index.skipped_large += 1;
                continue;
            }
            if index.files.len() >= limits.max_files {
                index.truncated_at_build = true;
                return Ok(());
            }
            let bytes = match fs::read(&path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            index.file_stamps.push(FileStamp {
                path: path.clone(),
                size_bytes: metadata.len(),
                modified_ns: modified_ns(&metadata),
                is_directory: false,
            });
            let path = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            let path_text = path.to_string_lossy().into_owned();
            let content = String::from_utf8_lossy(&bytes);
            let is_context_code = has_context_code_suffix(&path);
            let is_test_path = path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().to_ascii_lowercase().contains("test"));
            index.files.push(IndexedRepoFile {
                path,
                path_text,
                content: content.into_owned(),
                is_context_code,
                is_test_path,
            });
            let file_index = index.files.len() - 1;
            let terms = indexable_terms(&index.files[file_index].content);
            for term in terms {
                index
                    .term_files
                    .entry(term.clone())
                    .or_default()
                    .push(file_index);
                for trigram in term
                    .as_bytes()
                    .windows(3)
                    .map(|bytes| [bytes[0], bytes[1], bytes[2]])
                    .collect::<BTreeSet<_>>()
                {
                    let posting = index.term_trigram_files.entry(trigram).or_default();
                    if posting.last().copied() != Some(file_index) {
                        posting.push(file_index);
                    }
                }
                if is_context_code {
                    index
                        .context_term_files
                        .entry(term)
                        .or_default()
                        .push(file_index);
                }
            }
        }
    }
    Ok(())
}

fn modified_ns(metadata: &fs::Metadata) -> Option<u128> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_nanos())
}

fn has_skipped_suffix(path: &Path) -> bool {
    let display = path.to_string_lossy().to_ascii_lowercase();
    SKIP_SUFFIXES.iter().any(|suffix| display.ends_with(suffix))
}

fn has_context_code_suffix(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        let suffix = format!(".{}", extension.to_string_lossy().to_ascii_lowercase());
        CONTEXT_CODE_SUFFIXES.contains(&suffix.as_str())
    })
}

fn is_indexable_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn indexable_terms(content: &str) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    let mut start = None;
    for (offset, byte) in content.bytes().enumerate() {
        if byte.is_ascii_alphanumeric() || byte == b'_' {
            start.get_or_insert(offset);
        } else if let Some(begin) = start.take() {
            terms.insert(content[begin..offset].to_ascii_lowercase());
        }
    }
    if let Some(begin) = start {
        terms.insert(content[begin..].to_ascii_lowercase());
    }
    terms
}

fn truncate_line(line: &str, cap: usize) -> String {
    if line.len() <= cap {
        return line.to_owned();
    }
    let mut end = cap;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write(path: &Path, content: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut file = fs::File::create(path).unwrap();
        file.write_all(content).unwrap();
    }

    #[test]
    fn deterministic_search_skips_artifacts_and_respects_result_limit() {
        let root = tempdir().unwrap();
        write(&root.path().join("z.py"), b"needle z\nneedle z again\n");
        write(&root.path().join("a.rs"), b"needle a\n");
        write(&root.path().join("target/ignored.rs"), b"needle ignored\n");
        write(&root.path().join("model.safetensors"), b"needle binary\n");
        let result = search_repo(
            root.path(),
            "needle",
            RepoSearchLimits {
                max_results: 2,
                ..RepoSearchLimits::default()
            },
        )
        .unwrap();
        assert!(result.truncated);
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.matches[0].path, PathBuf::from("a.rs"));
        assert_eq!(result.matches[1].path, PathBuf::from("z.py"));
    }

    #[test]
    fn large_files_are_explicitly_skipped_and_empty_needles_refuse() {
        let root = tempdir().unwrap();
        write(&root.path().join("large.txt"), b"needle");
        let result = search_repo(
            root.path(),
            "needle",
            RepoSearchLimits {
                max_file_bytes: 1,
                ..RepoSearchLimits::default()
            },
        )
        .unwrap();
        assert_eq!(result.skipped_large, 1);
        assert!(result.matches.is_empty());
        assert!(search_repo(root.path(), "", RepoSearchLimits::default()).is_err());
    }

    #[test]
    fn bounded_file_read_preserves_hash_bytes_and_line_window_contract() {
        let root = tempdir().unwrap();
        let raw = b"alpha\r\nbeta\ngamma\n";
        write(&root.path().join("sample.py"), raw);

        let full = read_repo_file(root.path(), "sample.py", 64, None, None).unwrap();
        assert_eq!(full.bytes, raw.len() as u64);
        assert_eq!(full.content, "alpha\r\nbeta\ngamma\n");
        assert_eq!(full.sha256, sha256_hex(raw));
        assert!(!full.truncated);
        assert_eq!(full.shown_bytes, raw.len());
        assert!(full.start_line.is_none());

        let window = read_repo_file(root.path(), "sample.py", 64, Some(2), Some(3)).unwrap();
        assert_eq!(window.content, "beta\ngamma\n");
        assert_eq!(window.start_line, Some(2));
        assert_eq!(window.end_line, Some(3));
        assert_eq!(window.total_lines, Some(3));
        assert!(!window.out_of_range);

        let missing_window = read_repo_file(root.path(), "sample.py", 64, Some(9), None).unwrap();
        assert_eq!(missing_window.content, "");
        assert_eq!(missing_window.start_line, Some(9));
        assert_eq!(missing_window.end_line, Some(9));
        assert_eq!(missing_window.total_lines, Some(3));
        assert!(missing_window.out_of_range);
        assert!(missing_window.window_note.is_some());
    }

    #[test]
    fn bounded_file_read_bills_truncation_and_containment() {
        let root = tempdir().unwrap();
        write(&root.path().join("sample.py"), b"1234567890");
        let clipped = read_repo_file(root.path(), "sample.py", 4, None, None).unwrap();
        assert_eq!(clipped.content, "1234");
        assert_eq!(clipped.shown_bytes, 4);
        assert!(clipped.truncated);
        assert!(clipped
            .truncation_note
            .as_deref()
            .unwrap()
            .contains("4 of 10"));

        let outside = root.path().parent().unwrap().join("outside-hawking-file");
        write(&outside, b"no");
        assert!(read_repo_file(root.path(), &outside, 64, None, None).is_err());
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn cached_identifier_query_only_visits_indexed_files() {
        let root = tempdir().unwrap();
        write(&root.path().join("a.rs"), b"pub fn indexed_token() {}\n");
        write(&root.path().join("b.rs"), b"pub fn unrelated() {}\n");
        let index = RepoSearchIndex::build(root.path(), RepoSearchLimits::default()).unwrap();
        let result = index.search("indexed_token", 10).unwrap();
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].path, PathBuf::from("a.rs"));
        assert_eq!(index.search("INDEXED_TOKEN", 10).unwrap().matches.len(), 1);
        assert!(index.search("absent_token", 10).unwrap().matches.is_empty());
    }

    #[test]
    fn scoped_search_preserves_glob_case_and_per_file_bounds() {
        let root = tempdir().unwrap();
        write(
            &root.path().join("a.py"),
            b"needle one\nneedle two\nNEEDLE upper\n",
        );
        write(&root.path().join("b.rs"), b"needle rust\n");
        let index = RepoSearchIndex::build(root.path(), RepoSearchLimits::default()).unwrap();
        let result = index
            .search_files(Path::new(""), "needle", "*.py", 10, Some(1))
            .unwrap();
        assert_eq!(result.files_seen, 1);
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].line, 1);
        assert_eq!(result.matches[0].path, PathBuf::from("a.py"));
    }

    #[test]
    fn resident_service_swaps_only_complete_rebuilt_generations() {
        let root = tempdir().unwrap();
        write(&root.path().join("a.rs"), b"pub fn before_token() {}\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();
        assert_eq!(service.generation(), 1);
        assert_eq!(service.search("before_token", 10).unwrap().matches.len(), 1);
        write(&root.path().join("b.rs"), b"pub fn after_token() {}\n");
        assert!(service
            .search("after_token", 10)
            .unwrap()
            .matches
            .is_empty());
        let rebuilt = service.rebuild().unwrap();
        assert_eq!(rebuilt.generation, 2);
        assert_eq!(service.generation(), 2);
        assert_eq!(service.search("after_token", 10).unwrap().matches.len(), 1);
    }

    #[test]
    fn resident_path_rank_refreshes_only_when_directory_layout_changes() {
        let root = tempdir().unwrap();
        fs::create_dir(root.path().join("hcli")).unwrap();
        write(&root.path().join("hcli/serve.py"), b"def serve(): pass\n");
        write(&root.path().join("notes.txt"), b"not a code path\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();
        let suffixes = vec![".py".to_owned()];

        let first = service
            .rank_code_paths(&suffixes, &["hcli/serve.py".to_owned()], 4)
            .unwrap();
        assert_eq!(first.paths, vec![PathBuf::from("hcli/serve.py")]);
        assert_eq!(first.files_seen, 1);
        assert!(first.complete);
        assert_eq!(service.path_generation(), 1);

        write(&root.path().join("hcli/worker.py"), b"def worker(): pass\n");
        let second = service
            .rank_code_paths(&suffixes, &["worker.py".to_owned()], 4)
            .unwrap();
        assert_eq!(second.paths, vec![PathBuf::from("hcli/worker.py")]);
        assert_eq!(second.files_seen, 2);
        assert!(second.complete);
        assert_eq!(service.path_generation(), 2);
    }

    #[test]
    fn resident_content_rank_combines_terms_and_scopes_deterministically() {
        let root = tempdir().unwrap();
        fs::create_dir(root.path().join("hcli")).unwrap();
        fs::create_dir(root.path().join("tools")).unwrap();
        write(
            &root.path().join("hcli/session_transport.py"),
            b"streaming browser chat preserves durable session state\n",
        );
        write(
            &root.path().join("tools/session_probe.py"),
            b"streaming browser chat probe\n",
        );
        write(
            &root.path().join("README.md"),
            b"unrelated repository notes\n",
        );
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();

        let result = service
            .rank_content_paths(
                &[
                    "streaming".to_owned(),
                    "durable".to_owned(),
                    "session".to_owned(),
                ],
                Some("hcli/"),
                false,
                4,
            )
            .unwrap();
        assert_eq!(
            result.paths,
            vec![PathBuf::from("hcli/session_transport.py")]
        );
        assert_eq!(result.terms_used, 3);
        assert_eq!(result.files_seen, 2);
    }

    #[test]
    fn resident_content_rank_preserves_precomputed_test_focus() {
        let root = tempdir().unwrap();
        write(&root.path().join("hcli/target.py"), b"shared marker\n");
        write(&root.path().join("hcli/test_target.py"), b"shared marker\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();

        let result = service
            .rank_content_paths(&["shared".to_owned()], Some("hcli/"), true, 2)
            .unwrap();
        assert_eq!(
            result.paths,
            vec![
                PathBuf::from("hcli/test_target.py"),
                PathBuf::from("hcli/target.py")
            ]
        );
    }

    #[test]
    fn filesystem_search_rebuilds_when_source_metadata_changes() {
        let root = tempdir().unwrap();
        write(&root.path().join("a.py"), b"before_token\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();
        let initial = service
            .search_files(root.path(), "before_token", "*.py", 10, None)
            .unwrap();
        assert_eq!(initial.matches.len(), 1);
        write(&root.path().join("a.py"), b"after_token\n");
        let result = service
            .search_files(root.path(), "after_token", "*.py", 10, None)
            .unwrap();
        assert_eq!(result.matches.len(), 1);
        assert_eq!(service.generation(), 2);
        assert_eq!(service.path_generation(), 1);
    }

    #[test]
    fn filesystem_search_does_not_rebuild_for_unrelated_file_changes() {
        let root = tempdir().unwrap();
        write(&root.path().join("a.py"), b"needle\n");
        write(&root.path().join("b.rs"), b"unrelated\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();

        write(&root.path().join("b.rs"), b"unrelated_changed\n");
        let result = service
            .search_files(root.path(), "needle", "*.py", 10, None)
            .unwrap();

        assert_eq!(result.matches.len(), 1);
        assert_eq!(service.generation(), 1);
    }

    #[test]
    fn filesystem_search_preserves_substrings_inside_identifier_tokens() {
        let root = tempdir().unwrap();
        write(
            &root.path().join("a.py"),
            b"prefixneedle_suffix\nneedless\n",
        );
        write(&root.path().join("b.py"), b"needle\n");
        let service = RepoSearchService::open(root.path(), RepoSearchLimits::default()).unwrap();
        let index = service.index.read().unwrap();
        assert!(index
            .term_trigram_files
            .values()
            .all(|posting| posting.windows(2).all(|pair| pair[0] != pair[1])));
        drop(index);

        let result = service
            .search_files(root.path(), "needle", "*.py", 10, None)
            .unwrap();

        assert_eq!(
            result
                .matches
                .iter()
                .map(|item| (item.path.clone(), item.line))
                .collect::<Vec<_>>(),
            vec![
                (PathBuf::from("a.py"), 1),
                (PathBuf::from("a.py"), 2),
                (PathBuf::from("b.py"), 1),
            ]
        );
    }

    #[test]
    fn bounded_list_is_content_free_deterministic_and_contained() {
        let root = tempdir().unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        write(&root.path().join("b.rs"), b"secret body");
        write(&root.path().join("a.rs"), b"another body");
        write(&root.path().join("note.txt"), b"not selected");

        let result = list_repo_directory(root.path(), root.path(), "*", 10, false).unwrap();
        assert_eq!(result.directories_seen, 1);
        assert_eq!(
            result
                .files
                .iter()
                .map(|row| row.filename.as_str())
                .collect::<Vec<_>>(),
            ["a.rs", "b.rs", "note.txt"]
        );
        assert_eq!(result.directories[0].filename, "src");
        assert!(result.files.iter().all(|row| row.size_bytes.is_some()));
        assert!(result.files.iter().all(|row| row.entry_type == "file"));
        assert!(result
            .directories
            .iter()
            .all(|row| row.size_bytes.is_none()));
        assert!(result
            .files
            .iter()
            .all(|row| row.path.to_string_lossy() != "secret body"));
        assert!(!simple_glob_match("note.txt", "*.rs"));
        assert!(simple_glob_match("a.rs", "?.rs"));
    }

    #[test]
    fn bounded_list_refuses_paths_outside_native_root() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let error = list_repo_directory(root.path(), outside.path(), "*", 1, false)
            .expect_err("outside path must be refused");
        assert!(error.to_string().contains("outside the native root"));
    }
}
