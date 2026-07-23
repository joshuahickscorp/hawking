//! Wire hide-compat repo instructions into the compiled turn context (Bible sec
//! 78.1 #11 partial, sec 20).
//!
//! A repository migrated from another coding agent (Claude Code) carries a tree
//! of instruction files: `CLAUDE.md` memory with `@import`s, `.claude/CLAUDE.md`,
//! `CLAUDE.local.md`, and un-scoped `.claude/rules/*.md`. [`hide_compat`] reads
//! and resolves that tree WITHOUT a model (pure parsing + precedence). This module
//! folds the RESOLVED launch instruction order into the compiled `ContextPack` as
//! a pinned instruction/system context source so the migrated repo's house rules
//! ride into every turn, honoring precedence (read-last-wins: the more-specific
//! instruction is emitted LAST so it wins), and surfaces WHICH files loaded as a
//! context receipt (the retained span's provenance `derived_from`, plus a durable
//! `context.instructions` marker written by the turn core).
//!
//! The resolved set is loaded once at workspace open and cached on
//! [`crate::services::BackendServices`] so it is not re-parsed every turn.
//! Cache-invalidation on a live config-file change is a DEFERRED note (reopen the
//! workspace to pick up edits). A model-assisted reconciliation of conflicting
//! instructions is DEFERRED_MODEL_REQUIRED: this path never loads a model.

use std::collections::HashSet;
use std::path::Path;

use futures::future::BoxFuture;
use hawking_context::compiler::{CompileInput, ContextCandidate, ContextSource};
use hawking_context::manifest::{ContextSourceKind, PinState};
use hide_compat::layout::Layout;
use hide_compat::CompatConfig;
use hide_core::error::Result;
use hide_core::types::{Provenance, TrustLevel};

/// One instruction file that contributed to the compiled context, for the
/// context receipt. `bytes` is the resolved (injection-ready) body length.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadedInstruction {
    pub path: String,
    pub kind: String,
    pub bytes: usize,
}

/// The resolved, launch-ordered repo instructions plus a receipt of which files
/// contributed. Cached on the services so it is not re-parsed every turn.
#[derive(Debug, Clone, Default)]
pub struct ResolvedInstructions {
    /// The concatenated, provenance-headed instruction text in launch order
    /// (root first, more-specific last -> read-last-wins precedence).
    pub text: String,
    /// The files that contributed, in the same order (the context receipt).
    pub files: Vec<LoadedInstruction>,
}

impl ResolvedInstructions {
    /// No repo instructions (an un-migrated repo, or a load failure).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Whether there is nothing to inject.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() || self.text.trim().is_empty()
    }

    /// The loaded file paths, in launch order (the receipt as a path list).
    pub fn file_paths(&self) -> Vec<String> {
        self.files.iter().map(|f| f.path.clone()).collect()
    }

    /// The receipt as JSON, for a durable `context.instructions` marker event.
    pub fn receipt_json(&self) -> serde_json::Value {
        serde_json::json!({
            "files": self.files,
            "count": self.files.len(),
        })
    }

    /// Build the pinned instruction [`ContextSource`] carrying this resolved set.
    pub fn as_source(&self) -> CompatInstructionsSource {
        CompatInstructionsSource {
            text: self.text.clone(),
            files: self.file_paths(),
        }
    }
}

/// Resolve the launch instruction order for a [`Layout`] into a single, ordered,
/// provenance-headed block plus a receipt. A load failure (a malformed
/// settings.json, say) degrades to empty rather than failing the caller: missing
/// migration config must never break a turn.
pub fn resolve_repo_instructions(layout: &Layout) -> ResolvedInstructions {
    match CompatConfig::load(layout, None) {
        Ok(config) => resolve_from_config(&config),
        Err(_) => ResolvedInstructions::empty(),
    }
}

/// Resolve a repo's launch instructions from a workspace root. Repo-scoped by
/// design: `home` is pointed at an internal path that holds no `~/.claude` tree,
/// so ONLY the repository's own instruction files load. The user-global
/// `~/.claude/CLAUDE.md` is intentionally EXCLUDED here (folding a machine-global
/// file into every workspace turn is a separate concern, DEFERRED). `repo_root ==
/// cwd`, so subtree `CLAUDE.md` files stay lazy (they attach to matching-file
/// reads, not launch).
pub fn resolve_repo_instructions_for_root(workspace_root: &Path) -> ResolvedInstructions {
    let no_home = workspace_root.join(".hide").join("compat-no-user-home");
    let layout = Layout::new(workspace_root, workspace_root, no_home);
    resolve_repo_instructions(&layout)
}

/// Fold a loaded [`CompatConfig`] into launch-ordered instruction text + receipt.
/// Order + precedence come straight from [`CompatConfig::launch_instruction_order`]
/// (the authority): launch-injected memory (root first, more-specific last)
/// followed by un-scoped rules. read-last-wins means a later (more-specific)
/// instruction is emitted after an earlier one, so it wins.
fn resolve_from_config(config: &CompatConfig) -> ResolvedInstructions {
    use std::collections::HashMap;
    use std::path::PathBuf;

    // path -> (kind label, injection-ready body). Memory entries carry their
    // comment-stripped, import-inlined `injected` body; un-scoped rules carry
    // their frontmatter-stripped `body`.
    let mut by_path: HashMap<PathBuf, (String, String)> = HashMap::new();
    for entry in config.memory.launch_entries() {
        by_path.insert(
            entry.path.clone(),
            (
                memory_kind_label(entry.kind).to_string(),
                entry.injected.clone(),
            ),
        );
    }
    for rule in &config.rules {
        if rule.loads_at_launch() {
            by_path
                .entry(rule.path.clone())
                .or_insert_with(|| ("rule".to_string(), rule.body.clone()));
        }
    }

    let mut text = String::new();
    let mut files = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for path in config.launch_instruction_order() {
        if !seen.insert(path.clone()) {
            continue;
        }
        let (kind, body) = match by_path.get(&path) {
            Some(v) => v,
            None => continue,
        };
        let body = body.trim_end_matches(['\n', '\r']).trim_end();
        if body.trim().is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        // A provenance header names the source file + precedence layer so the
        // model can attribute a rule and later layers read as overrides.
        text.push_str(&format!("# Instruction ({kind}): {}\n", path.display()));
        text.push_str(body);
        files.push(LoadedInstruction {
            path: path.display().to_string(),
            kind: kind.clone(),
            bytes: body.len(),
        });
    }

    ResolvedInstructions { text, files }
}

fn memory_kind_label(kind: hide_compat::claude_md::MemoryKind) -> &'static str {
    use hide_compat::claude_md::MemoryKind::*;
    match kind {
        UserGlobal => "user-global",
        Project => "project",
        DotClaude => "dot-claude",
        Local => "local",
        Subtree => "subtree",
    }
}

/// A pinned, never-evict instruction context source carrying the resolved repo
/// instructions. It contributes exactly ONE span (the whole launch-ordered
/// block) so the precedence order is preserved VERBATIM in the compiled prompt
/// (multiple pinned spans would be re-ordered by the head/tail packer). The
/// span's provenance `derived_from` lists every loaded file, so the compiled
/// manifest's retained span IS the context receipt.
pub struct CompatInstructionsSource {
    text: String,
    files: Vec<String>,
}

impl ContextSource for CompatInstructionsSource {
    fn name(&self) -> &str {
        "compat_instructions"
    }

    fn gather<'a>(
        &'a self,
        _input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        Box::pin(async move {
            if self.text.trim().is_empty() {
                return Ok(Vec::new());
            }
            let mut c = ContextCandidate::new(
                "compat_instructions:0",
                ContextSourceKind::System,
                "Repository instructions (Claude Code compat)",
                self.text.clone(),
                1.0,
                Provenance {
                    source: "compat_instructions".to_string(),
                    trust: TrustLevel::Trusted,
                    confidence: 1.0,
                    labels: vec!["instructions".to_string(), "claude-md-compat".to_string()],
                    derived_from: self.files.clone(),
                },
            );
            // Pin to the head, never evicted: repo house rules always ride.
            c.pin = PinState::NeverEvict;
            c.importance = Some(1.0);
            Ok(vec![c])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_context::compiler::{CompileInput, ContextCompiler};
    use hawking_context::manifest::ContextSourceKind;
    use hawking_context::profiles::ContextProfile;
    use hide_core::ids::{now_ms, ModelId};
    use hide_core::runtime::{ModelArchitecture, ModelDescriptor};

    fn model(ctx: usize) -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId::new(),
            name: "test".to_string(),
            architecture: ModelArchitecture::Transformer,
            context_tokens: ctx,
            tokenizer_signature: "test".to_string(),
            footprint_mb: 1,
        }
    }

    /// A throwaway temp repo with a `.git` marker + whatever instruction files a
    /// test writes, and a private `home` that holds no `~/.claude` tree (so only
    /// repo-local files load). Returns the repo root; the caller writes files.
    fn temp_repo(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("hide_compat_instr_{tag}_{}", now_ms()));
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join(".hide")).unwrap();
        root
    }

    #[test]
    fn resolves_root_claude_md_into_ordered_text_with_receipt() {
        let root = temp_repo("root");
        std::fs::write(
            root.join("CLAUDE.md"),
            "# House rules\nRULE_ALPHA_TOKEN: always run cargo test before commit.\n",
        )
        .unwrap();

        let resolved = resolve_repo_instructions_for_root(&root);
        assert!(!resolved.is_empty(), "root CLAUDE.md must resolve");
        // The distinctive token from the CLAUDE.md rode into the resolved text.
        assert!(
            resolved.text.contains("RULE_ALPHA_TOKEN"),
            "resolved text must carry the CLAUDE.md rule, got: {}",
            resolved.text
        );
        // The receipt names the file that loaded.
        assert!(
            resolved
                .files
                .iter()
                .any(|f| f.path.ends_with("CLAUDE.md") && f.kind == "project"),
            "receipt must list the project CLAUDE.md, got: {:?}",
            resolved.files
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn instruction_token_rides_into_compiled_context_pack() {
        let root = temp_repo("compiled");
        std::fs::write(
            root.join("CLAUDE.md"),
            "COMPILED_RULE_TOKEN: prefer small, reversible increments.\n",
        )
        .unwrap();
        let resolved = resolve_repo_instructions_for_root(&root);

        let mut compiler = ContextCompiler::new();
        compiler.add_source(resolved.as_source());
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(1024),
                model: model(4096),
                task: "unrelated task text".to_string(),
            })
            .await
            .unwrap();

        // The pinned instruction span rides into the compiled prompt...
        assert!(
            compiled.prompt.contains("COMPILED_RULE_TOKEN"),
            "instruction must ride into the compiled prompt, got: {}",
            compiled.prompt
        );
        // ...and the retained span's provenance IS the context receipt (it names
        // the loaded file in `derived_from`).
        let span = compiled
            .manifest
            .retained
            .iter()
            .find(|s| s.provenance.source == "compat_instructions")
            .expect("compat instruction span retained");
        assert_eq!(span.source, ContextSourceKind::System);
        assert!(
            span.provenance.derived_from.iter().any(|p| p.ends_with("CLAUDE.md")),
            "receipt (derived_from) must list the loaded file, got: {:?}",
            span.provenance.derived_from
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn precedence_more_specific_instruction_is_read_last() {
        // Root CLAUDE.md (project) + a nested CLAUDE.md one directory deeper. With
        // cwd deeper than repo_root, the nested file is a launch-injected Project
        // entry read AFTER the root one (root first, more-specific last), so it
        // wins under read-last-wins. Also add an un-scoped rule, which loads at
        // launch AFTER all memory.
        let root = temp_repo("prec");
        let sub = root.join("service");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            root.join("CLAUDE.md"),
            "ROOT_LAYER_TOKEN: default indent is four spaces.\n",
        )
        .unwrap();
        std::fs::write(
            sub.join("CLAUDE.md"),
            "NESTED_LAYER_TOKEN: this service overrides indent to two spaces.\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join(".claude").join("rules")).unwrap();
        std::fs::write(
            root.join(".claude").join("rules").join("general.md"),
            "RULE_LAYER_TOKEN: prefer explicit error handling.\n",
        )
        .unwrap();

        // cwd = sub (deeper than repo_root = root), private home (no user tree).
        let no_home = root.join(".hide").join("compat-no-user-home");
        let layout = Layout::new(&root, &sub, &no_home);
        let resolved = resolve_repo_instructions(&layout);

        let root_at = resolved
            .text
            .find("ROOT_LAYER_TOKEN")
            .expect("root layer present");
        let nested_at = resolved
            .text
            .find("NESTED_LAYER_TOKEN")
            .expect("nested layer present");
        let rule_at = resolved
            .text
            .find("RULE_LAYER_TOKEN")
            .expect("rule layer present");
        // More-specific (nested) is read AFTER the root layer -> wins last.
        assert!(
            root_at < nested_at,
            "the more-specific nested instruction must be read last (win); text: {}",
            resolved.text
        );
        // Un-scoped rules load at launch AFTER all memory.
        assert!(
            nested_at < rule_at,
            "un-scoped rules load after memory; text: {}",
            resolved.text
        );
        // All three files are on the receipt.
        assert_eq!(resolved.files.len(), 3, "receipt: {:?}", resolved.files);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn un_migrated_repo_resolves_empty() {
        let root = temp_repo("empty");
        let resolved = resolve_repo_instructions_for_root(&root);
        assert!(resolved.is_empty());
        assert!(resolved.files.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
