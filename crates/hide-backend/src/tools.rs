use crate::security::SecurityServices;
use hide_core::config::HideConfig;
use hide_core::tool::ToolDispatcher;
use hide_core::tool::ToolRegistry;
use std::sync::Arc;

pub fn build_default_tool_registry() -> ToolRegistry {
    let registry = ToolRegistry::default();
    hide_tools::register_builtin_tools(&registry);
    registry
}

pub fn build_default_tool_dispatcher(
    config: &HideConfig,
    registry: Arc<ToolRegistry>,
) -> ToolDispatcher {
    ToolDispatcher::new(
        registry,
        Arc::new(SecurityServices::permission_engine(config)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::tool::ToolCall;
    use hide_core::types::Decision;
    use serde_json::json;

    #[test]
    fn default_registry_contains_builtin_tools() {
        let registry = build_default_tool_registry();
        let names: Vec<_> = registry.specs().into_iter().map(|spec| spec.name).collect();
        assert!(names.contains(&"fs.read".to_string()));
        assert!(names.contains(&"fs.write".to_string()));
        assert!(names.contains(&"shell.plan".to_string()));
    }

    #[tokio::test]
    async fn dispatcher_uses_workspace_policy_for_writes() {
        let dir =
            std::env::temp_dir().join(format!("hide_backend_tools_{}", hide_core::ids::now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let registry = Arc::new(build_default_tool_registry());
        let dispatcher = build_default_tool_dispatcher(&config, registry);
        let file = dir.join("allowed.txt");

        let result = dispatcher
            .dispatch(ToolCall::new(
                "fs.write",
                json!({
                    "path": file.to_string_lossy(),
                    "content": "allowed",
                    "create_dirs": true
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, hide_core::tool::ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "allowed");
        let _ = std::fs::remove_dir_all(dir);
    }
}
