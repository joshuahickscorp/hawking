//! hide-protocol: the single schema authority for the HIDE Agent Server.
//!
//! HIDE's agent server has ONE semantic object model (Bible sec 14) and ONE
//! wire protocol (Bible sec 15). This crate is the sole place both are defined,
//! in Rust, with serde AND schemars derived from the SAME type definitions, so
//! the JSON Schema is generated from the code and can never silently drift from
//! it. Every other HIDE crate that speaks the protocol derives its shapes from
//! here rather than re-declaring them.
//!
//! Two halves:
//!
//! - [`model`], [`plan`], [`item`]: the semantic object model. Workspace ->
//!   Repository / Environment -> Session -> Thread -> Turn -> Item, plus Goal,
//!   Plan (a DAG of steps), Agent, Artifact, Checkpoint, StateCapsuleRef, Tool,
//!   and Oracle.
//! - [`protocol`]: the methods ([`Method`]), the server pushes
//!   ([`Notification`]), the request/response envelope, and the initialize
//!   handshake with capability negotiation.
//!
//! [`compat`] reconciles this authority with `hide-core`'s current
//! `Intent`/`UiEvent` transport (Wire-B) so the elevation is provable in both
//! directions without refactoring `hide-core`.
//!
//! # Model-free
//!
//! This crate is schema-only and entirely model-free (RIP doctrine). It defines
//! and validates shapes over fixtures. It never runs a model, opens a network
//! connection, drives a browser, or produces live runtime bytes. The legs that
//! inherently need a running model -- binding a [`ModelPolicy`] to weights, and
//! producing the bytes a [`StateCapsuleRef`] points at -- are marked
//! `DEFERRED_MODEL_REQUIRED` at their definitions and are not implemented or
//! claimed here.
//!
//! ```
//! use hide_protocol::{Method, PROTOCOL_VERSION};
//!
//! assert_eq!(PROTOCOL_VERSION, "hide.agent.v1");
//! assert_eq!(Method::ThreadFork.as_str(), "thread/fork");
//! assert_eq!(Method::ThreadFork.namespace(), "thread");
//! ```

pub mod command;
pub mod compat;
pub mod error;
pub mod ids;
pub mod item;
pub mod model;
pub mod plan;
pub mod protocol;

pub use command::{
    command_catalog, ApprovalPolicy, BackendBinding, Category, CommandSpec, EffectClass,
    RequiredSelection, Surface, UndoStrategy, HOST_CAPABILITIES, INTENT_NAMES, WIRE_CUSTOM_NAMES,
};
pub use error::{ProtocolError, Result};
pub use item::{Item, ItemKind};
pub use model::{
    Artifact, ArtifactKind, Checkpoint, CompletionStatus, Environment, EnvironmentKind, Oracle,
    OracleKind, Repository, Risk, Session, SessionStatus, StateCapsuleRef, Thread, Tool, Turn,
    TurnRole, TurnStatus, VcsKind, Workspace,
};
pub use plan::{
    Agent, Budget, ContextScope, Cost, DagError, Effect, Goal, ModelPolicy, Plan, PlanStep,
    ReturnPolicy, RollbackBoundary, RollbackKind, Scope,
};
pub use protocol::{
    negotiate_capabilities, negotiate_version, ClientCapabilities, InitializeRequest,
    InitializeResult, Method, NegotiatedCapabilities, Notification, PeerInfo, Request, Response,
    RpcError, ServerCapabilities, PROTOCOL_VERSION,
};

/// Generate the JSON Schema for a model type as a [`serde_json::Value`].
///
/// This is the "one schema source" in action: the schema is derived from the
/// Rust type, not maintained by hand. Callers (codegen, contract tests, a
/// published schema bundle) render every wire type from here.
pub fn json_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T>();
    serde_json::to_value(root).expect("a schemars RootSchema always serializes to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::*;
    use crate::item::*;
    use crate::model::*;
    use crate::plan::*;
    use crate::protocol::*;
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use std::collections::BTreeSet;

    fn round_trip<T>(value: T)
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(&value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, value, "value must survive a serde_json round trip");
    }

    // -- fixture builders --------------------------------------------------

    fn sample_scope() -> Scope {
        Scope {
            paths: vec!["src/retry.ts".into()],
            network: false,
            description: Some("the retry module only".into()),
        }
    }

    fn sample_plan() -> Plan {
        Plan {
            id: PlanId::from("pln_1"),
            goal: Some(GoalId::from("goal_1")),
            steps: vec![
                PlanStep {
                    id: StepId::from("stp_1"),
                    objective: "reproduce the flake".into(),
                    dependencies: vec![],
                    scope: sample_scope(),
                    effects: vec![Effect::ReadFs, Effect::Shell],
                    expected_artifacts: vec!["repro.log".into()],
                    acceptance_oracle: Some(OracleId::from("orc_test")),
                    rollback_boundary: RollbackBoundary::default(),
                    cost: Cost {
                        tokens: Some(1200),
                        wall_ms: Some(30_000),
                        usd_micros: None,
                    },
                    parallelizable: false,
                },
                PlanStep {
                    id: StepId::from("stp_2"),
                    objective: "apply the fix".into(),
                    dependencies: vec![StepId::from("stp_1")],
                    scope: sample_scope(),
                    effects: vec![Effect::WriteFs],
                    expected_artifacts: vec![],
                    acceptance_oracle: None,
                    rollback_boundary: RollbackBoundary {
                        kind: RollbackKind::Checkpoint,
                        reference: Some("ckpt_pre".into()),
                    },
                    cost: Cost::default(),
                    parallelizable: true,
                },
            ],
            created_ms: 1,
        }
    }

    fn sample_state_capsule_ref() -> StateCapsuleRef {
        StateCapsuleRef {
            id: StateCapsuleId::from("cap_1"),
            capsule_id: "hs_cap_abc".into(),
            digest: "blake3:deadbeef".into(),
            model_id: Some("mdl_x".into()),
            size_bytes: Some(4096),
            created_ms: 2,
        }
    }

    fn sample_artifact() -> Artifact {
        Artifact {
            id: ArtifactId::from("art_1"),
            name: "patch.diff".into(),
            kind: ArtifactKind::Patch,
            path: Some("out/patch.diff".into()),
            digest: Some("sha256:abc".into()),
            size_bytes: Some(128),
            produced_by: Some(StepId::from("stp_2")),
            created_ms: 3,
        }
    }

    fn sample_checkpoint() -> Checkpoint {
        Checkpoint {
            id: CheckpointId::from("ckpt_1"),
            session: Some(SessionId::from("ses_1")),
            thread: Some(ThreadId::from("thr_1")),
            at_turn: Some(TurnId::from("trn_1")),
            label: Some("pre-fix".into()),
            capsule: Some(sample_state_capsule_ref()),
            vcs_ref: Some("stash@{0}".into()),
            created_ms: 4,
        }
    }

    /// One item per ItemKind, in the order the model lists them, so a coverage
    /// test can prove every kind is present.
    fn one_item_per_kind() -> Vec<ItemKind> {
        vec![
            ItemKind::UserMessage(UserMessage {
                text: "hello".into(),
                attachments: vec![Attachment {
                    id: "blb_1".into(),
                    hash: "h".into(),
                    size_bytes: 3,
                    media_type: None,
                }],
            }),
            ItemKind::AgentMessage(AgentMessage {
                text: "on it".into(),
            }),
            ItemKind::ReasoningSummary(ReasoningSummary {
                text: "considering three fixes".into(),
            }),
            ItemKind::Plan(sample_plan()),
            ItemKind::PlanMutation(PlanMutation {
                plan: PlanId::from("pln_1"),
                kind: PlanMutationKind::AddStep,
                step: Some(StepId::from("stp_3")),
                detail: None,
            }),
            ItemKind::ContextReceipt(ContextReceipt {
                sources: vec![ContextSource {
                    name: "retry.ts".into(),
                    uri: Some("file://retry.ts".into()),
                    trust: Some("workspace".into()),
                    token_estimate: Some(300),
                }],
                total_token_estimate: Some(300),
                note: None,
            }),
            ItemKind::ToolCall(ToolCall {
                call_id: ToolCallId::from("tcl_1"),
                tool: ToolId::from("tool_bash"),
                arguments: serde_json::json!({ "cmd": "cargo test" }),
            }),
            ItemKind::ToolResult(ToolResult {
                call_id: ToolCallId::from("tcl_1"),
                ok: true,
                output: serde_json::json!({ "code": 0 }),
                error: None,
            }),
            ItemKind::ShellStream(ShellStream {
                call_id: Some(ToolCallId::from("tcl_1")),
                channel: ShellChannel::Stdout,
                chunk: "running 12 tests".into(),
            }),
            ItemKind::Patch(Patch {
                patch_id: "p1".into(),
                summary: Some("fix retry".into()),
                files: vec!["retry.ts".into()],
                unified_diff: "--- a\n+++ b\n".into(),
            }),
            ItemKind::Diff(Diff {
                diff_id: "d1".into(),
                path: "retry.ts".into(),
                hunks: vec![DiffHunk {
                    old_start: 1,
                    old_lines: 2,
                    new_start: 1,
                    new_lines: 3,
                    text: "@@".into(),
                }],
                status: DiffStatus::Proposed,
            }),
            ItemKind::ApprovalRequest(ApprovalRequest {
                request_id: ApprovalId::from("apr_1"),
                action: "write retry.ts".into(),
                risk: Risk::Low,
                effects: vec![Effect::WriteFs],
                detail: None,
            }),
            ItemKind::ApprovalResult(ApprovalResult {
                request_id: ApprovalId::from("apr_1"),
                decision: ApprovalDecision::Approved,
                reason: None,
            }),
            ItemKind::VerificationRequest(VerificationRequest {
                request_id: VerificationId::from("ver_1"),
                oracle: OracleId::from("orc_test"),
                target: Some("stp_2".into()),
            }),
            ItemKind::VerificationReceipt(VerificationReceipt {
                request_id: VerificationId::from("ver_1"),
                oracle: OracleId::from("orc_test"),
                outcome: VerificationOutcome::Pass,
                detail: None,
            }),
            ItemKind::Artifact(sample_artifact()),
            ItemKind::Checkpoint(sample_checkpoint()),
            ItemKind::StateCapsule(sample_state_capsule_ref()),
            ItemKind::AgentSpawn(AgentSpawn {
                agent: AgentId::from("agt_1"),
                role: "fixer".into(),
                objective: "fix the flake".into(),
            }),
            ItemKind::AgentResult(AgentResult {
                agent: AgentId::from("agt_1"),
                outcome: CompletionStatus::Success,
                summary: Some("green".into()),
            }),
            ItemKind::Steer(Steer {
                directive: "keep the one that passes".into(),
            }),
            ItemKind::Interrupt(Interrupt {
                reason: Some("stop".into()),
            }),
            ItemKind::Error(ErrorItem {
                code: "E1".into(),
                message: "boom".into(),
            }),
            ItemKind::Completion(Completion {
                status: CompletionStatus::Success,
                summary: None,
            }),
            ItemKind::Blocker(Blocker {
                code: "B1".into(),
                message: "need a token".into(),
                needs: Some("api key".into()),
            }),
        ]
    }

    fn sample_turn() -> Turn {
        let items = one_item_per_kind()
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Item::new(ItemId::from(format!("itm_{i}")), i as u64, kind))
            .collect();
        Turn {
            id: TurnId::from("trn_1"),
            thread: ThreadId::from("thr_1"),
            role: TurnRole::Agent,
            status: TurnStatus::Running,
            items,
            parent_turn: None,
            created_ms: 5,
        }
    }

    fn sample_thread() -> Thread {
        Thread {
            id: ThreadId::from("thr_1"),
            session: SessionId::from("ses_1"),
            parent: Some(ThreadId::from("thr_0")),
            forked_at_turn: Some(TurnId::from("trn_0")),
            ephemeral: true,
            title: Some("branch A".into()),
            turns: vec![sample_turn()],
            created_ms: 6,
        }
    }

    fn sample_session() -> Session {
        Session {
            id: SessionId::from("ses_1"),
            workspace: WorkspaceId::from("wsp_1"),
            repository: Some(RepositoryId::from("repo_1")),
            environment: Some(EnvironmentId::from("env_1")),
            title: Some("auth retry".into()),
            threads: vec![ThreadId::from("thr_1")],
            status: SessionStatus::Active,
            created_ms: 7,
        }
    }

    fn sample_agent() -> Agent {
        Agent {
            id: AgentId::from("agt_1"),
            role: "fixer".into(),
            parent: Some(AgentId::from("agt_root")),
            children: vec![AgentId::from("agt_2")],
            model_policy: ModelPolicy {
                policy: "default".into(),
                tier: Some("frontier".into()),
                allow_fallback: true,
            },
            context_scope: ContextScope {
                include: vec!["src/**".into()],
                exclude: vec!["secrets/**".into()],
                inherit_parent: true,
            },
            effects: vec![Effect::ReadFs, Effect::WriteFs, Effect::Shell],
            budget: Budget {
                tokens: Some(50_000),
                wall_ms: Some(120_000),
                tool_calls: Some(40),
                usd_micros: None,
            },
            return_policy: ReturnPolicy::StateCapsule,
        }
    }

    // -- round-trip every top type ----------------------------------------

    #[test]
    fn every_top_type_round_trips_through_serde_json() {
        let workspace = Workspace {
            id: WorkspaceId::from("wsp_1"),
            name: "hawking".into(),
            repositories: vec![Repository {
                id: RepositoryId::from("repo_1"),
                workspace: WorkspaceId::from("wsp_1"),
                name: "hawking".into(),
                root_path: "/repo".into(),
                vcs: VcsKind::Git,
                remote_url: Some("https://example/hawking".into()),
                head_ref: Some("main".into()),
            }],
            environments: vec![Environment {
                id: EnvironmentId::from("env_1"),
                workspace: WorkspaceId::from("wsp_1"),
                name: "local".into(),
                kind: EnvironmentKind::Local,
                working_dir: "/repo".into(),
                platform: Some("darwin".into()),
                capabilities: vec!["shell".into()],
            }],
            sessions: vec![SessionId::from("ses_1")],
            default_environment: Some(EnvironmentId::from("env_1")),
            created_ms: 0,
        };
        round_trip(workspace.clone());
        round_trip(workspace.repositories[0].clone());
        round_trip(workspace.environments[0].clone());
        round_trip(sample_session());
        round_trip(sample_thread());
        round_trip(sample_turn());
        for kind in one_item_per_kind() {
            round_trip(Item::new(ItemId::from("itm_x"), 0, kind));
        }
        round_trip(Goal {
            id: GoalId::from("goal_1"),
            session: Some(SessionId::from("ses_1")),
            statement: "keep the fix that passes".into(),
            acceptance: vec!["tests green".into()],
            acceptance_oracle: Some(OracleId::from("orc_test")),
            constraints: vec!["touch only retry.ts".into()],
            created_ms: 0,
        });
        round_trip(sample_plan());
        round_trip(sample_artifact());
        round_trip(sample_checkpoint());
        round_trip(sample_state_capsule_ref());
        round_trip(sample_agent());
        round_trip(Tool {
            id: ToolId::from("tool_bash"),
            name: "bash".into(),
            description: Some("run a shell command".into()),
            effects: vec![Effect::Shell],
            input_schema: serde_json::json!({ "type": "object" }),
            output_schema: Some(serde_json::json!({ "type": "string" })),
            requires_approval: true,
        });
        round_trip(Oracle {
            id: OracleId::from("orc_test"),
            name: "cargo test".into(),
            kind: OracleKind::Test,
            command: Some(vec!["cargo".into(), "test".into()]),
            acceptance: "exit code 0".into(),
            deterministic: true,
        });
        round_trip(Method::ThreadFork);
        round_trip(Notification::ItemAdded {
            item: Item::new(
                ItemId::from("itm_1"),
                0,
                ItemKind::AgentMessage(AgentMessage { text: "hi".into() }),
            ),
        });
        round_trip(Request {
            id: RequestId::from("req_1"),
            method: Method::TurnCreate,
            params: serde_json::json!({ "text": "go" }),
        });
        round_trip(Response {
            id: RequestId::from("req_1"),
            result: Some(serde_json::json!({ "ok": true })),
            error: None,
        });
        round_trip(sample_initialize_request());
        round_trip(sample_initialize_result());
    }

    // -- schema generation -------------------------------------------------

    #[test]
    fn json_schema_generates_for_the_core_wire_types() {
        for schema in [
            json_schema::<Session>(),
            json_schema::<Thread>(),
            json_schema::<Turn>(),
            json_schema::<Item>(),
            json_schema::<Method>(),
            json_schema::<Notification>(),
            json_schema::<InitializeRequest>(),
            json_schema::<InitializeResult>(),
            json_schema::<Plan>(),
            json_schema::<Agent>(),
        ] {
            assert!(schema.is_object(), "each generated schema is a JSON object");
            assert!(
                schema.get("$schema").is_some() || schema.get("title").is_some(),
                "a schemars root schema carries a $schema or title marker"
            );
        }
    }

    // -- ItemKind coverage -------------------------------------------------

    #[test]
    fn item_kind_covers_every_listed_kind() {
        let expected: BTreeSet<&str> = [
            "user_message",
            "agent_message",
            "reasoning_summary",
            "plan",
            "plan_mutation",
            "context_receipt",
            "tool_call",
            "tool_result",
            "shell_stream",
            "patch",
            "diff",
            "approval_request",
            "approval_result",
            "verification_request",
            "verification_receipt",
            "artifact",
            "checkpoint",
            "state_capsule",
            "agent_spawn",
            "agent_result",
            "steer",
            "interrupt",
            "error",
            "completion",
            "blocker",
        ]
        .into_iter()
        .collect();

        let built: BTreeSet<&str> = one_item_per_kind().iter().map(|k| k.tag()).collect();
        assert_eq!(built, expected, "every listed item kind is constructible");

        // And each tag is what the serialized item actually carries.
        for kind in one_item_per_kind() {
            let tag = kind.tag();
            let item = Item::new(ItemId::from("itm_x"), 0, kind);
            let value = serde_json::to_value(&item).unwrap();
            assert_eq!(
                value.get("kind").and_then(|v| v.as_str()),
                Some(tag),
                "the flattened item carries its kind tag on the wire"
            );
        }
    }

    // -- initialize handshake shape (sec 15.3) -----------------------------

    fn sample_initialize_request() -> InitializeRequest {
        InitializeRequest {
            client: PeerInfo {
                name: "hide-desktop".into(),
                version: "0.2.2".into(),
            },
            protocol_versions: vec![PROTOCOL_VERSION.to_string()],
            capabilities: ClientCapabilities {
                streaming: true,
                approvals: true,
                fs: true,
                terminal: true,
                subscriptions: true,
                experimental: Default::default(),
            },
        }
    }

    fn sample_initialize_result() -> InitializeResult {
        InitializeResult {
            server: PeerInfo {
                name: "hide-serve".into(),
                version: "0.2.2".into(),
            },
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities::full(),
        }
    }

    #[test]
    fn initialize_request_matches_sec_15_3_shape() {
        let value = serde_json::to_value(sample_initialize_request()).unwrap();
        // client { name, version }
        let client = value.get("client").expect("client object");
        assert_eq!(client.get("name").unwrap(), "hide-desktop");
        assert_eq!(client.get("version").unwrap(), "0.2.2");
        // protocolVersions is a camelCase array of version strings
        let versions = value
            .get("protocolVersions")
            .and_then(|v| v.as_array())
            .expect("protocolVersions array");
        assert_eq!(versions[0], PROTOCOL_VERSION);
        // capabilities present
        assert!(value.get("capabilities").is_some(), "capabilities present");
    }

    #[test]
    fn initialize_result_carries_single_negotiated_version() {
        let value = serde_json::to_value(sample_initialize_result()).unwrap();
        assert_eq!(value.get("protocolVersion").unwrap(), PROTOCOL_VERSION);
        assert!(value.get("server").is_some());
        assert!(value.get("capabilities").is_some());
    }

    #[test]
    fn version_negotiation_picks_the_highest_shared_by_server_preference() {
        let server = vec!["hide.agent.v2".to_string(), "hide.agent.v1".to_string()];
        let client = vec!["hide.agent.v1".to_string()];
        assert_eq!(
            negotiate_version(&client, &server),
            Some("hide.agent.v1".to_string())
        );
        assert_eq!(negotiate_version(&["hide.agent.v9".to_string()], &server), None);
    }

    #[test]
    fn capability_negotiation_ands_shared_flags() {
        let client = ClientCapabilities {
            streaming: true,
            approvals: true,
            fs: false,
            terminal: false,
            subscriptions: false,
            experimental: Default::default(),
        };
        let server = ServerCapabilities::full();
        let effective = negotiate_capabilities(&client, &server);
        assert!(effective.streaming, "both want streaming");
        assert!(
            !effective.subscriptions,
            "client did not opt into subscriptions"
        );
        assert!(effective.state, "server-only capability passes through");
    }

    // -- method namespace coverage -----------------------------------------

    #[test]
    fn methods_cover_every_required_namespace() {
        let namespaces: BTreeSet<&str> = Method::ALL.iter().map(|m| m.namespace()).collect();
        for ns in [
            "workspace",
            "environment",
            "session",
            "thread",
            "goal",
            "turn",
            "item",
            "agent",
            "checkpoint",
            "state",
            "approval",
            "artifact",
        ] {
            assert!(namespaces.contains(ns), "missing method namespace: {ns}");
        }
        // The distinctive verbs the Bible calls out explicitly are present.
        let all: BTreeSet<&str> = Method::ALL.iter().map(|m| m.as_str()).collect();
        for m in [
            "thread/fork",
            "thread/fork_ephemeral",
            "thread/merge_summary",
            "turn/steer",
            "turn/interrupt",
            "turn/pause",
            "turn/resume",
            "item/subscribe",
            "state/save",
            "state/load",
            "state/fork",
            "state/release",
            "state/inspect",
        ] {
            assert!(all.contains(m), "missing method: {m}");
        }
    }

    #[test]
    fn method_serializes_as_its_slash_string() {
        let value = serde_json::to_value(Method::StateFork).unwrap();
        assert_eq!(value, serde_json::json!("state/fork"));
        let back: Method = serde_json::from_value(serde_json::json!("thread/merge_summary")).unwrap();
        assert_eq!(back, Method::ThreadMergeSummary);
    }

    // -- plan DAG ----------------------------------------------------------

    #[test]
    fn plan_dag_validation_accepts_acyclic_rejects_cycles_and_dangling() {
        let good = sample_plan();
        assert!(good.validate_dag().is_ok(), "sample plan is a valid DAG");

        let mut cyclic = sample_plan();
        // stp_1 -> stp_2 already; make stp_1 depend on stp_2 to close a cycle.
        cyclic.steps[0].dependencies = vec![StepId::from("stp_2")];
        assert_eq!(cyclic.validate_dag(), Err(DagError::Cycle));

        let mut dangling = sample_plan();
        dangling.steps[0].dependencies = vec![StepId::from("stp_missing")];
        assert!(matches!(
            dangling.validate_dag(),
            Err(DagError::DanglingDependency { .. })
        ));

        let mut dup = sample_plan();
        dup.steps[1].id = StepId::from("stp_1");
        assert!(matches!(
            dup.validate_dag(),
            Err(DagError::DuplicateStepId(_))
        ));
    }

    #[test]
    fn notification_method_tag_matches_serialized_form() {
        let n = Notification::TurnStarted {
            turn: TurnId::from("trn_1"),
        };
        let value = serde_json::to_value(&n).unwrap();
        assert_eq!(value.get("method").unwrap(), n.method());
        assert_eq!(value.get("method").unwrap(), "turn/started");
    }
}
