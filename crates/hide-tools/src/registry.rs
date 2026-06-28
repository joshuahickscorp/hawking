//! Registration of the builtin catalog into a `hide-core` `ToolRegistry`.

use crate::edit::{ApplyPatchTool, SearchReplaceTool, WriteFileTool};
use crate::fs::{FsConfig, FsGlobTool, FsListTool, FsReadTool, FsStatTool, FsWatchTool, FsWriteTool};
use crate::git::{
    GitCommitTool, GitConfig, GitDiffTool, GitLogTool, GitStatusTool, GitWorktreeAddTool,
    GitWorktreeListTool, GitWorktreeRemoveTool,
};
use crate::proc::ProcTool;
use crate::search::SearchTextTool;
use crate::shell::{ShellConfig, ShellPlanTool, ShellRunTool};
use hide_core::tool::ToolRegistry;

/// Register the full builtin catalog with default (no-blob, default-sandbox)
/// configuration.
pub fn register_builtin_tools(registry: &ToolRegistry) {
    register_builtin_tools_with(registry, ShellConfig::default());
}

/// Register the full builtin catalog, threading a [`ShellConfig`] (workspace root,
/// blob store for large-output spill, sandbox toggle) into the process and FS
/// tools. The blob store, when present, is shared so over-cap output spills to the
/// same CAS the event log uses.
pub fn register_builtin_tools_with(registry: &ToolRegistry, shell_config: ShellConfig) {
    let fs_config = FsConfig {
        blobs: shell_config.blobs.clone(),
    };
    let git_config = GitConfig {
        blobs: shell_config.blobs.clone(),
    };

    // Filesystem
    registry.register(FsReadTool::with_config(fs_config.clone()));
    registry.register(FsListTool::default());
    registry.register(FsWriteTool::default());
    registry.register(FsStatTool::default());
    registry.register(FsGlobTool::default());
    registry.register(FsWatchTool::default());

    // Edit family
    registry.register(SearchReplaceTool::default());
    registry.register(ApplyPatchTool::default());
    registry.register(WriteFileTool::default());

    // Shell / process
    registry.register(ShellRunTool::with_config(shell_config.clone()));
    registry.register(ShellPlanTool::with_config(shell_config.clone()));
    registry.register(ProcTool::test_run().with_config(shell_config.clone()));
    registry.register(ProcTool::build_run().with_config(shell_config.clone()));
    registry.register(ProcTool::compile_check().with_config(shell_config.clone()));

    // Search
    registry.register(SearchTextTool::default());

    // Git
    registry.register(GitStatusTool::with_config(git_config.clone()));
    registry.register(GitDiffTool::with_config(git_config.clone()));
    registry.register(GitLogTool::with_config(git_config.clone()));
    registry.register(GitCommitTool::with_config(git_config.clone()));
    registry.register(GitWorktreeAddTool::with_config(git_config.clone()));
    registry.register(GitWorktreeRemoveTool::with_config(git_config.clone()));
    registry.register(GitWorktreeListTool::with_config(git_config));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_full_catalog() {
        let registry = ToolRegistry::default();
        register_builtin_tools(&registry);
        let names: Vec<String> = registry.specs().into_iter().map(|s| s.name).collect();
        for expected in [
            "fs.read",
            "fs.list",
            "fs.write",
            "fs.stat",
            "fs.glob",
            "fs.watch",
            "edit.search_replace",
            "edit.apply_patch",
            "edit.write_file",
            "shell.run",
            "shell.plan",
            "test.run",
            "build.run",
            "compile.check",
            "search.text",
            "git.status",
            "git.diff",
            "git.log",
            "git.commit",
            "git.worktree.add",
            "git.worktree.remove",
            "git.worktree.list",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
    }
}
