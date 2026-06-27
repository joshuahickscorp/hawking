use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::replay::BackendReplayService;
use crate::security::SecurityServices;
use crate::services::{BackendCapabilities, BackendServices, SharedBackend};
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use hide_core::api::{Intent, IntentAck, UiEvent};
use hide_core::event::{EventPayload, EventSource, NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{RunId, SessionId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::ModelRole;
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_kernel::machine::state::AgentState;
use hide_kernel::session::SessionProjection;
use hide_kernel::AgentKernel;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

pub struct BackendHost {
    pub services: SharedBackend,
    pub connectors: Arc<ConnectorRegistry>,
    pub tools: Arc<ToolRegistry>,
    pub dispatcher: ToolDispatcher,
    pub security: SecurityServices,
    pub replay: BackendReplayService,
    commands: CommandRouter,
    kernel: AgentKernel,
}

impl BackendHost {
    pub fn open_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::from_services(BackendServices::open_workspace(workspace_root)?)
    }

    pub fn from_services(services: BackendServices) -> Result<Self> {
        let services = Arc::new(services);
        let tools = Arc::new(build_default_tool_registry());
        let dispatcher = build_default_tool_dispatcher(&services.config, tools.clone());
        let connectors = Arc::new(ConnectorRegistry::default());
        register_backend_connectors(&connectors, &services);
        Ok(Self {
            commands: CommandRouter::new(services.event_log.clone()),
            kernel: AgentKernel::new(services.event_log.clone()),
            replay: BackendReplayService::new(
                services.event_log.clone(),
                services.projection_store.clone(),
            ),
            services,
            connectors,
            tools,
            dispatcher,
            security: SecurityServices::default(),
        })
    }

    pub async fn handle_intent(&self, intent: Intent) -> Result<IntentAck> {
        self.commands.handle(intent).await
    }

    pub async fn call_connector(&self, id: &str, method: &str, params: Value) -> Result<Value> {
        self.connectors.call(id, method, params).await
    }

    pub async fn rebuild_session_projection(
        &self,
        session_id: SessionId,
    ) -> Result<SessionProjection> {
        self.replay.rebuild_session(session_id).await
    }

    pub async fn ui_events(
        &self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<UiEvent>> {
        self.replay.ui_events(session_id, after_seq, limit).await
    }

    pub async fn run_command(
        &self,
        session_id: SessionId,
        argv: Vec<String>,
        cwd: Option<String>,
    ) -> Result<ToolResult> {
        let mut args = json!({ "argv": argv });
        if let Some(cwd) = cwd {
            args["cwd"] = json!(cwd);
        }
        self.dispatch_tool(
            session_id,
            None,
            ToolCall {
                id: hide_core::ids::ToolCallId::new(),
                tool_name: "shell.run".to_string(),
                args,
                capability_grant_id: None,
                idempotency_key: None,
                dry_run: false,
            },
        )
        .await
    }

    pub async fn dispatch_tool(
        &self,
        session_id: SessionId,
        run_id: Option<RunId>,
        call: ToolCall,
    ) -> Result<ToolResult> {
        let call_event = call.clone();
        let result = self.dispatcher.dispatch(call).await?;
        self.services
            .event_log
            .append(NewEvent {
                session_id: session_id.clone(),
                run_id: run_id.clone(),
                parent: None,
                source: EventSource::Agent,
                kind: "tool.call".into(),
                payload: EventPayload::ToolCall(ToolCallEvent {
                    call_id: call_event.id,
                    tool_name: call_event.tool_name,
                    capability_grant_id: call_event.capability_grant_id,
                    args: call_event.args,
                    predicted_effects: result.effects.clone(),
                }),
                redactions: Vec::new(),
            })
            .await?;
        let result_event = self
            .services
            .event_log
            .append(NewEvent {
                session_id,
                run_id,
                parent: None,
                source: EventSource::Tool,
                kind: "tool.result".into(),
                payload: EventPayload::ToolResult(ToolResultEvent {
                    call_id: result.call_id.clone(),
                    ok: result.status == ToolStatus::Ok,
                    summary: tool_result_summary(&result),
                    output: result.structured_content.clone(),
                    bytes_ref: result.bytes_ref.clone(),
                }),
                redactions: Vec::new(),
            })
            .await?;
        self.services.projection_store.put_projection(
            &result_event.session_id,
            result_event.seq,
            json!({
                "projection": "last_tool_result",
                "tool_status": result.status,
                "tool_output": result.structured_content.clone(),
            }),
        )?;
        Ok(result)
    }

    pub async fn run_agent_to_terminal(
        &self,
        session_id: SessionId,
        objective: impl Into<String>,
        max_steps: usize,
    ) -> Result<AgentState> {
        let mut state = self.kernel.start_run(session_id, objective).await?;
        for _ in 0..max_steps {
            if state.phase.is_terminal() {
                break;
            }
            self.kernel.step(&mut state).await?;
        }
        Ok(state)
    }

    pub async fn status(&self) -> BackendStatus {
        BackendStatus {
            workspace_root: self.services.config.workspace_root.clone(),
            capabilities: self.services.capabilities.clone(),
            connectors: self.connectors.statuses().await,
            tools: self.tools.specs(),
            model_roles: self.services.role_registry.all(),
        }
    }

    pub async fn health(&self) -> HealthReport {
        let mut checks = Vec::new();
        let layout = self.services.layout();
        checks.push(path_check("hide_dir", &layout.hide_dir));
        checks.push(path_check("event_log", &layout.event_log));
        checks.push(path_check("blobs", &layout.blobs));
        checks.push(path_check("projections", &layout.projections));
        checks.push(path_check("kv", &layout.kv));
        checks.push(count_check("tools", self.tools.specs().len()));
        checks.push(count_check(
            "model_roles",
            self.services.role_registry.all().len(),
        ));
        for connector in self.connectors.statuses().await {
            checks.push(HealthCheck {
                name: format!("connector:{}", connector.id),
                status: if connector.healthy {
                    HealthStatus::Ok
                } else {
                    HealthStatus::Failed
                },
                detail: connector.detail,
            });
        }
        let status = if checks
            .iter()
            .any(|check| check.status == HealthStatus::Failed)
        {
            HealthStatus::Failed
        } else if checks
            .iter()
            .any(|check| check.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        };
        HealthReport {
            component: "hide-backend".to_string(),
            status,
            checks,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub workspace_root: PathBuf,
    pub capabilities: BackendCapabilities,
    pub connectors: Vec<ConnectorStatus>,
    pub tools: Vec<ToolSpec>,
    pub model_roles: Vec<ModelRole>,
}

fn tool_result_summary(result: &ToolResult) -> String {
    if let Some(error) = &result.error {
        return format!("{}: {}", error.code, error.message);
    }
    if let Some(value) = &result.structured_content {
        return value.to_string();
    }
    format!("{:?}", result.status)
}

fn path_check(name: &str, path: &std::path::Path) -> HealthCheck {
    let exists = path.exists();
    HealthCheck {
        name: name.to_string(),
        status: if exists {
            HealthStatus::Ok
        } else {
            HealthStatus::Failed
        },
        detail: if exists {
            path.display().to_string()
        } else {
            format!("missing {}", path.display())
        },
    }
}

fn count_check(name: &str, count: usize) -> HealthCheck {
    HealthCheck {
        name: name.to_string(),
        status: if count == 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        },
        detail: count.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_research::{ResearchRun, ResearchState};
    use hide_core::api::UiEventKind;
    use hide_core::config::HideConfig;
    use hide_core::event::EventPayload;
    use hide_core::ids::{now_ms, ToolCallId};
    use hide_core::tool::ToolCall;
    use hide_core::types::Decision;

    #[tokio::test]
    async fn host_dispatches_tool_and_records_events() {
        let dir = std::env::temp_dir().join(format!("hide_host_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session_id = host.services.session();
        let file = dir.join("host.txt");

        let result = host
            .dispatch_tool(
                session_id.clone(),
                None,
                ToolCall {
                    id: ToolCallId::new(),
                    tool_name: "fs.write".to_string(),
                    args: json!({
                        "path": file.to_string_lossy(),
                        "content": "host write",
                        "create_dirs": true
                    }),
                    capability_grant_id: None,
                    idempotency_key: None,
                    dry_run: false,
                },
            )
            .await
            .unwrap();

        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "host write");
        let events = host
            .services
            .event_log
            .scan(Some(session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(events
            .iter()
            .any(|event| matches!(event.payload, EventPayload::ToolCall(_))));
        assert!(events
            .iter()
            .any(|event| matches!(event.payload, EventPayload::ToolResult(_))));
        assert!(host
            .services
            .projection_store
            .latest_projection(&session_id)
            .unwrap()
            .is_some());
        let ui_events = host
            .ui_events(Some(session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(ui_events
            .iter()
            .any(|event| matches!(event.kind, UiEventKind::ToolProgress { .. })));
        let rebuilt = host
            .rebuild_session_projection(session_id.clone())
            .await
            .unwrap();
        assert_eq!(rebuilt.session_id, session_id);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_reports_status_surface() {
        let dir = std::env::temp_dir().join(format!("hide_host_status_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let status = host.status().await;
        assert!(status.capabilities.agent_kernel);
        assert!(status.tools.iter().any(|tool| tool.name == "fs.write"));
        assert!(status
            .connectors
            .iter()
            .any(|connector| connector.id == "research"));
        assert!(status
            .model_roles
            .iter()
            .any(|role| role.name == "hawking-hero-coder"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_records_run_command_intent_and_executes_command_api() {
        let dir = std::env::temp_dir().join(format!("hide_host_command_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.shell_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();

        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["printf".to_string(), "intent".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted);

        let session_id = host.services.session();
        let result = host
            .run_command(
                session_id,
                vec!["printf".to_string(), "api".to_string()],
                None,
            )
            .await
            .unwrap();

        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(result.structured_content.unwrap()["stdout"], "api");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_routes_connector_calls() {
        let dir = std::env::temp_dir().join(format!("hide_host_connector_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut run = ResearchRun::new("host connector");
        run.state = ResearchState::Complete;

        host.call_connector("research", "runs.append", json!({ "run": run }))
            .await
            .unwrap();
        let listed = host
            .call_connector("research", "runs.list", json!({ "limit": 1 }))
            .await
            .unwrap();

        assert_eq!(listed["runs"].as_array().unwrap().len(), 1);
        assert_eq!(listed["runs"][0]["topic"], "host connector");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_reports_health_checks() {
        let dir = std::env::temp_dir().join(format!("hide_host_health_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let health = host.health().await;

        assert_eq!(health.status, HealthStatus::Ok);
        assert!(health.checks.iter().any(|check| check.name == "tools"));
        assert!(health
            .checks
            .iter()
            .any(|check| check.name == "connector:personalization"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
