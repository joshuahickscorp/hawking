use crate::fs::{FsListTool, FsReadTool, FsWriteTool};
use crate::git::GitStatusTool;
use crate::shell::{ShellPlanTool, ShellRunTool};
use hide_core::tool::ToolRegistry;

pub fn register_builtin_tools(registry: &ToolRegistry) {
    registry.register(FsReadTool::default());
    registry.register(FsListTool::default());
    registry.register(FsWriteTool::default());
    registry.register(ShellPlanTool::default());
    registry.register(ShellRunTool::default());
    registry.register(GitStatusTool::default());
}
