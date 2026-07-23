//! hide-compat: configuration-compatibility readers.
//!
//! HIDE is a local-first IDE. A repository that was set up for another coding
//! agent (Claude Code, in particular) carries a tree of configuration files:
//! `CLAUDE.md` memory with `@imports`, `.claude/rules`, `settings.json` across
//! several scopes, subagent and skill definitions, and `.mcp.json` server
//! declarations. This crate reads all of that WITHOUT running any model. It is
//! pure parsing plus precedence resolution so a repo migrates to HIDE with
//! minimal changes (Bible sec 58 / Phase 11).
//!
//! Every reader takes a [`layout::Layout`], which names the filesystem locations
//! for each scope. Nothing here touches process globals beyond the optional
//! `Layout::discover` helper, so the whole crate is deterministic and fully
//! testable against a tempdir fixture.
//!
//! The two precedence rules that most often trip people up are kept explicit:
//! - Settings permission arrays MERGE across scopes and `deny` wins a decision;
//!   scalar settings resolve Managed > CLI > Local > Project > User.
//! - Instruction layers use a SEPARATE precedence Managed > User > Project >
//!   Local, applied read-last-wins.

pub mod agents;
pub mod claude_md;
pub mod error;
pub mod frontmatter;
pub mod layout;
pub mod mcp;
pub mod rules;
pub mod settings;
pub mod skills;

pub use error::{CompatError, Result};
pub use layout::Layout;

/// The complete compatibility view of a repository.
#[derive(Debug, Clone)]
pub struct CompatConfig {
    pub memory: claude_md::MemoryTree,
    pub rules: Vec<rules::Rule>,
    pub settings: settings::ResolvedSettings,
    pub agents: Vec<agents::Agent>,
    pub skills: Vec<skills::Skill>,
    pub mcp: mcp::McpConfig,
}

impl CompatConfig {
    /// Read and resolve every compatibility source for a layout.
    ///
    /// `cli` optionally supplies the CLI settings scope (parsed command-line
    /// flags), which sits just under Managed in scalar precedence.
    pub fn load(layout: &Layout, cli: Option<settings::RawSettings>) -> Result<CompatConfig> {
        let settings = settings::load(layout, cli)?;
        let excludes = settings.excludes_glob_set()?;
        let memory = claude_md::discover(layout, excludes.as_ref());
        let rules = rules::discover(layout);
        let agents = agents::discover(layout);
        let skills = skills::discover(layout);
        let mcp = mcp::load(layout)?;

        Ok(CompatConfig {
            memory,
            rules,
            settings,
            agents,
            skills,
            mcp,
        })
    }

    /// The instruction sources injected at launch, in application order:
    /// memory files (root first, more specific last) followed by un-scoped
    /// rules. Scoped rules and lazy subtree memory are excluded (they attach to
    /// matching-file reads instead).
    pub fn launch_instruction_order(&self) -> Vec<std::path::PathBuf> {
        let mut order: Vec<std::path::PathBuf> = self.memory.launch_order();
        for rule in &self.rules {
            if rule.loads_at_launch() {
                order.push(rule.path.clone());
            }
        }
        order
    }
}
