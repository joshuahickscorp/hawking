use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::interrupt::InterruptHub;
use crate::replay::BackendReplayService;
use crate::security::SecurityServices;
use crate::services::{BackendCapabilities, BackendServices, SharedBackend};
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use crate::ui_bus::UiEventBus;
use hide_core::api::{Intent, IntentAck, UiEvent, UiEventKind};
use hide_core::event::{NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{RunId, SessionId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::ModelRole;
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_fleet::manager::KernelRunLauncher;
use hide_fleet::{
    AgentJob, ConcurrencyClass, FixedResourceProbe, FleetConfig, FleetGovernor, FleetManager,
    PriorityClass, ResourceSnapshot,
};
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
    kernel: Arc<AgentKernel>,
    /// The push Wire-B bus (broadcast + coalescing). The pull `ui_events` API is
    /// retained for replay/catch-up; this is the live path.
    ui_bus: Arc<UiEventBus>,
    /// Shared with the CommandRouter so control intents reach running runs.
    interrupts: Arc<InterruptHub>,
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
        let interrupts = Arc::new(InterruptHub::default());
        Ok(Self {
            commands: CommandRouter::with_interrupts(
                services.event_log.clone(),
                interrupts.clone(),
            ),
            kernel: Arc::new(AgentKernel::new(services.event_log.clone())),
            replay: BackendReplayService::new(
                services.event_log.clone(),
                services.projection_store.clone(),
            ),
            services,
            connectors,
            tools,
            dispatcher,
            security: SecurityServices::default(),
            ui_bus: Arc::new(UiEventBus::default()),
            interrupts,
        })
    }

    /// Subscribe to the live push UiEvent stream (Wire-B). Ordered; a lagging
    /// subscriber gets a `Lagged` signal rather than stalling the host.
    pub fn subscribe_ui(&self) -> tokio::sync::broadcast::Receiver<UiEvent> {
        self.ui_bus.subscribe()
    }

    /// The push UiEvent bus (for callers that want to publish/coalesce directly).
    pub fn ui_bus(&self) -> &Arc<UiEventBus> {
        &self.ui_bus
    }

    /// The interrupt hub control intents signal onto (shared with the kernel).
    pub fn interrupts(&self) -> &Arc<InterruptHub> {
        &self.interrupts
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
        self.dispatch_tool(session_id, None, ToolCall::new("shell.run", args))
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
        let mut call_new = NewEvent::tool_call(
            session_id.clone(),
            ToolCallEvent {
                call_id: call_event.call_id,
                tool_name: call_event.tool,
                capability_grant_id: call_event.capability_grant_id,
                args: call_event.args,
                predicted_effects: result.effects.clone(),
            },
        );
        call_new.run_id = run_id.clone();
        let call_event_record = self.services.event_log.append(call_new).await?;
        // The tool.result Observation pairs back to the tool.call Action via
        // `cause` (T3 Action/Observation replay pairing).
        let mut result_new = NewEvent::tool_result(
            session_id,
            ToolResultEvent {
                call_id: result.call_id.clone(),
                ok: result.status == ToolStatus::Ok,
                summary: tool_result_summary(&result),
                output: result.structured_content.clone(),
                bytes_ref: result.bytes_ref.clone(),
            },
        );
        result_new.run_id = run_id;
        result_new.cause = Some(call_event_record.id);
        let result_event = self.services.event_log.append(result_new).await?;
        self.services.projection_store.put_projection(
            &result_event.session_id,
            result_event.seq,
            json!({
                "projection": "last_tool_result",
                "tool_status": result.status,
                "tool_output": result.structured_content.clone(),
            }),
        )?;
        // Push the tool progress onto the live Wire-B bus (in addition to the
        // durable log the pull API replays from).
        self.ui_bus.publish(UiEvent {
            seq: result_event.seq,
            session_id: Some(result_event.session_id.clone()),
            kind: UiEventKind::ToolProgress {
                call_id: result.call_id.as_str().to_string(),
                message: if result.status == ToolStatus::Ok {
                    tool_result_summary(&result)
                } else {
                    format!("failed: {}", tool_result_summary(&result))
                },
            },
        });
        Ok(result)
    }

    /// Schedule a parallel kernel run via `hide_fleet::FleetManager` and drive it
    /// to completion (the now-real fleet path — the previously-dead `hide-fleet`
    /// dep is load-bearing here). The run is enqueued, admitted under the fleet
    /// Governor, isolated in a (fake-git, in this shell) worktree, and driven by a
    /// `KernelRunLauncher` over the host's kernel. Returns the job's terminal
    /// status string.
    ///
    /// `provider` is optional: when `Some`, the kernel is built with an HTTP
    /// `ModelProvider`-backed runtime so the fleet run generates against a live
    /// (or fake) serve; when `None`, the host's minimal stub kernel runs.
    pub async fn fleet_run(&self, session_id: SessionId, objective: impl Into<String>) -> Result<String> {
        // A deterministic fixed probe with ample headroom (no thermal/RAM
        // pressure) so the run admits in the test/headless path; production swaps
        // in `OsResourceProbe`.
        let probe = Arc::new(FixedResourceProbe {
            snapshot: ResourceSnapshot {
                free_memory_mb: 32_768,
                ..ResourceSnapshot::idle()
            },
        });
        let launcher = Arc::new(KernelRunLauncher::new(self.kernel.clone()).with_max_steps(64));
        let manager = FleetManager::new(
            self.services.event_log.clone(),
            FleetGovernor::default(),
            probe,
            launcher,
            FleetConfig::default(),
        )
        .with_fake_worktrees();

        let job = AgentJob::new(objective, PriorityClass::Normal)
            .with_session(session_id)
            .with_concurrency_class(ConcurrencyClass::Model);
        let job_id = job.id.clone();
        manager.enqueue(job).await?;
        manager.run_to_quiescence(2, 64).await?;

        let status = manager
            .queue()
            .get(&job_id)
            .map(|j| format!("{:?}", j.status))
            .unwrap_or_else(|| "Unknown".to_string());
        Ok(status)
    }

    /// Generate against a (supervised) runtime through the kernel's runtime-client
    /// seam and publish the completion onto the push Wire-B bus.
    ///
    /// This is the host's end-to-end generation path: a `KernelRuntimeClient`
    /// (router + the host's HTTP `ModelProvider`, adapted to the orch
    /// `InferenceClient` seam) produces tokens; each token batch is published —
    /// with coalescing — onto the broadcast bus, then flushed at stream end. The
    /// returned string is the full completion (for callers that also want it
    /// inline). `base_url` is the supervised serve's base (from the
    /// `RuntimeSupervisor`).
    pub async fn generate_and_publish(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<String> {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::router::SimpleRouter;
        use hide_core::runtime::{InferenceRequest, StreamChunk};
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let provider = HttpModelProvider::new(base_url);
        let inference = Arc::new(ModelProviderInferenceClient::new(provider));
        let router = Arc::new(SimpleRouter::new(self.services.role_registry.clone()));
        let runtime = KernelRuntimeClient::new(router, inference);

        let request = InferenceRequest {
            task_kind: "code".to_string(),
            prompt: prompt.into(),
            messages: Vec::new(),
            max_output_tokens: 256,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: Default::default(),
        };
        // Record a runtime.status event so the stream has a stable seq to key the
        // published UiEvent off of.
        let status_event = self
            .services
            .event_log
            .append(NewEvent::system(
                session_id.clone(),
                "runtime.generation",
                json!({ "task": "code" }),
            ))
            .await?;
        let stream_id = status_event.seq.to_string();

        let mut buf = String::new();
        {
            let bus = self.ui_bus.clone();
            let sess = session_id.clone();
            let sid = stream_id.clone();
            let seq = status_event.seq;
            let mut sink = |chunk: StreamChunk| {
                match chunk {
                    StreamChunk::Token { text, .. } => {
                        buf.push_str(&text);
                        // Push each token batch onto the bus (coalesced per stream).
                        bus.publish_token(seq, Some(sess.clone()), &sid, &text);
                    }
                    StreamChunk::Done { .. } => {
                        // Flush the coalesced batch at stream end.
                        bus.flush(Some(sess.clone()));
                    }
                    StreamChunk::Error { message } => {
                        bus.publish(UiEvent {
                            seq,
                            session_id: Some(sess.clone()),
                            kind: UiEventKind::Error {
                                code: "generation".to_string(),
                                message,
                            },
                        });
                    }
                }
                Ok(())
            };
            runtime.generate(request, &mut sink).await?;
        }
        Ok(buf)
    }

    /// Time-travel: scrub a session's projection to (and including) `seq`. A
    /// read-only view into the past (does not clobber the live projection).
    pub async fn scrub_to_event(
        &self,
        session_id: SessionId,
        seq: u64,
    ) -> Result<SessionProjection> {
        self.replay.scrub_to_event(session_id, seq).await
    }

    /// Time-travel: fork a new session from `from`'s log prefix up to `at_seq`.
    pub async fn fork_session(
        &self,
        from: SessionId,
        at_seq: u64,
    ) -> Result<(SessionId, SessionProjection)> {
        self.replay.fork_session(from, at_seq).await
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
    use hide_core::ids::now_ms;
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
                ToolCall::new(
                    "fs.write",
                    json!({
                        "path": file.to_string_lossy(),
                        "content": "host write",
                        "create_dirs": true
                    }),
                ),
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
        assert!(events.iter().any(|event| event.kind == "tool.call"));
        assert!(events.iter().any(|event| event.kind == "tool.result"));
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

    #[tokio::test]
    async fn host_caps_are_honest_remote_is_false() {
        let dir = std::env::temp_dir().join(format!("hide_host_caps_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let caps = host.status().await.capabilities;
        // Everything wired is true; the un-wired remote protocol is false.
        assert!(caps.agent_kernel && caps.fleet && caps.model_orchestration);
        assert!(!caps.remote_protocol);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_fleet_run_schedules_and_completes() {
        let dir = std::env::temp_dir().join(format!("hide_host_fleet_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        // Schedule a parallel kernel run via FleetManager; the minimal stub
        // kernel drives to Done. The previously-dead hide-fleet dep is now live.
        let status = host.fleet_run(session, "scaffold a module").await.unwrap();
        assert_eq!(status, "Done");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// THE FLAGSHIP integration test (WP-11). Proves the whole host loop:
    ///
    /// 1. Boot the [`RuntimeSupervisor`] against a FAKE in-process serve (health
    ///    + generate/embed stub) → state machine reaches `Ready`.
    /// 2. Drive an `Intent` through [`CommandRouter`] — it is *validated* and
    ///    accepted (a blank one would be rejected).
    /// 3. Generate through the kernel's runtime-client seam, backed by the HTTP
    ///    `ModelProvider` pointed at the supervised fake serve.
    /// 4. Assert the completion is published as a `UiEvent` on the broadcast bus
    ///    (the real Wire-B), with the text the fake runtime returned.
    ///
    /// This is the end-to-end path the audit said never closed: "the runtime is
    /// never booted; nothing flows end-to-end." It now flows.
    #[tokio::test]
    async fn flagship_boot_supervise_intent_generate_publish() {
        use crate::supervisor::testkit::{FakeLauncher, FakeRuntime};
        use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
        use hide_core::supervision::{BackoffPolicy, ProcessSpec};
        use std::time::Duration;

        let dir = std::env::temp_dir().join(format!("hide_flagship_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();

        // (1) Boot the supervisor against the fake serve.
        let rt = Arc::new(FakeRuntime::spawn().await);
        let cfg = SupervisorConfig {
            spec: ProcessSpec {
                name: "fake-serve".to_string(),
                argv: vec!["fake".to_string()],
                cwd: None,
                env: Default::default(),
                health_url: None,
            },
            backoff: BackoffPolicy::default(),
            health_interval: Duration::from_millis(10),
            boot_timeout: Duration::from_secs(2),
            lock_path: Some(host.services.layout().hide_dir.join("runtime.lock")),
        };
        let supervisor =
            RuntimeSupervisor::new(cfg, Arc::new(FakeLauncher::new(rt.clone())));
        supervisor.boot().await.unwrap();
        assert_eq!(
            supervisor.state(),
            hide_core::runtime::RuntimeSupervisorState::Ready
        );
        let base_url = supervisor.base_url().unwrap();

        // (2) Drive a validated intent through the command router.
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "implement the parser".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "valid SubmitTurn must be accepted");

        // (3+4) Subscribe to Wire-B, then generate against the supervised runtime
        // through the kernel runtime-client + HTTP ModelProvider, and assert the
        // completion is published on the broadcast bus.
        let mut rx = host.subscribe_ui();
        let completion = host
            .generate_and_publish(session.clone(), &base_url, "write a function")
            .await
            .unwrap();
        assert_eq!(completion, "fake generate");

        // The coalesced TokenBatch lands on the broadcast channel.
        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should be published")
            .expect("broadcast channel delivers");
        match event.kind {
            UiEventKind::TokenBatch { text, .. } => assert_eq!(text, "fake generate"),
            other => panic!("expected a TokenBatch UiEvent, got {other:?}"),
        }

        supervisor.shutdown().await;
        rt.stop();
        let _ = std::fs::remove_dir_all(dir);
    }
}
