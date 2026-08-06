//! hide-sdk: the generated client SDK and codegen for the HIDE Agent Server.
//!
//! Bible sec 15.7 states the rule this crate exists to enforce: "One source
//! must generate Rust types, TypeScript types, JSON Schema, OpenAPI
//! projections, protocol documentation, compatibility tests, event fixtures. No
//! handwritten frontend mirror types." That one source is `hide-protocol`.
//! hide-sdk reads its schemars-derived schemas and projects them, so nothing
//! downstream re-declares the protocol by hand.
//!
//! Three surfaces, all model-free:
//!
//! - [`schema`]: emit the protocol JSON Schema bundle from the ONE source.
//! - [`ts`]: a deterministic JSON-Schema-to-TypeScript emitter. The frontend's
//!   `.d.ts` types come from here, not from a handwritten mirror.
//! - [`client`]: a thin async client over a [`client::Transport`] trait, with a
//!   [`client::MockTransport`] for tests and typed helper methods that build
//!   `hide-protocol` [`Method`](crate::Method) requests and parse typed
//!   results.
//! - [`fixtures`]: canonical Notification/Item JSON fixtures the compatibility
//!   tests round-trip through `hide-protocol` serde.
//!
//! # Model-free
//!
//! Everything here is deterministic codegen and in-memory transport plumbing
//! over fixtures. It never runs a model or opens a socket. The real transport
//! that carries these requests to a live server is DEFERRED_MODEL_REQUIRED
//! -adjacent (a running agent server is required to exercise it end to end) and
//! is deliberately out of scope; see [`client::Transport`] for the seam and
//! [`client::MockTransport`] for the deterministic stand-in used in tests.

pub mod client {
    //! A thin async client over a [`Transport`].
    //!
    //! The client builds `hide-protocol` [`Request`] envelopes around typed
    //! [`Method`] values, sends them through a [`Transport`], and decodes the typed
    //! result. The typed helpers ([`Client::session_start`], [`Client::turn_start`],
    //! [`Client::item_subscribe`], [`Client::thread_fork`]) are the ergonomic
    //! surface a frontend or an integration would call.
    //!
    //! # DEFERRED_MODEL_REQUIRED
    //!
    //! The [`Transport`] trait is the seam to a live agent server. The real
    //! loopback / HTTP transport that carries these requests to a running
    //! `hide-serve` and streams notifications back needs a model-bearing server to
    //! answer them, so it is out of scope here and not implemented. Everything in
    //! this crate exercises the client over [`MockTransport`], a deterministic
    //! in-memory stand-in with a fixed routing table and a preloaded notification
    //! queue. No socket is opened and no model is run.

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde::de::DeserializeOwned;
    use serde_json::Value;
    use thiserror::Error;

    use crate::ids::RequestId;
    use crate::protocol::{Method, Notification, Request, Response, RpcError};

    /// Errors the SDK surfaces to a caller.
    #[derive(Debug, Error)]
    pub enum SdkError {
        /// The transport itself failed (connection, encode, timeout). The real
        /// transport is DEFERRED_MODEL_REQUIRED; the mock never returns this.
        #[error("transport error: {0}")]
        Transport(String),

        /// The server answered with a protocol-level error envelope.
        #[error("server returned an error for {method}: {code} {message}")]
        Rpc {
            method: String,
            code: i32,
            message: String,
        },

        /// The response envelope carried neither a result nor an error.
        #[error("no result in the response to {method}")]
        MissingResult { method: String },

        /// The result JSON did not decode into the expected typed result.
        #[error("could not decode the result for {method}: {source}")]
        Decode {
            method: String,
            #[source]
            source: serde_json::Error,
        },

        /// The mock transport had no canned response registered for the method.
        #[error("no mock handler registered for method {0}")]
        Unhandled(String),
    }

    /// The transport seam: send one [`Request`] and await its [`Response`], and
    /// drain any [`Notification`]s the server has pushed.
    ///
    /// DEFERRED_MODEL_REQUIRED: the production implementation talks to a live agent
    /// server. Only [`MockTransport`] is provided here.
    #[async_trait]
    pub trait Transport: Send + Sync {
        /// Send a request and await its response envelope.
        async fn request(&self, request: Request) -> Result<Response, SdkError>;

        /// Drain and return any notifications buffered for the client. A real
        /// streaming transport delivers these as they arrive; the mock returns
        /// whatever was preloaded.
        async fn notifications(&self) -> Vec<Notification>;
    }

    /// The typed client. Generic over the [`Transport`] so tests use
    /// [`MockTransport`] and production would use the deferred real transport.
    pub struct Client<T: Transport> {
        transport: T,
        next_id: AtomicU64,
    }

    impl<T: Transport> Client<T> {
        /// Wrap a transport. Request ids are minted deterministically starting at
        /// `req_1`, so a fixed sequence of calls produces a fixed set of ids.
        pub fn new(transport: T) -> Self {
            Self {
                transport,
                next_id: AtomicU64::new(1),
            }
        }

        /// Borrow the underlying transport (handy for asserting what the mock saw).
        pub fn transport(&self) -> &T {
            &self.transport
        }

        fn mint_id(&self) -> RequestId {
            let n = self.next_id.fetch_add(1, Ordering::SeqCst);
            RequestId::from(format!("req_{n}"))
        }

        /// The core call: build a [`Request`] around a [`Method`] and params, send
        /// it, and decode the typed result `R`. Every typed helper routes through
        /// here.
        pub async fn call<R: DeserializeOwned>(
            &self,
            method: Method,
            params: Value,
        ) -> Result<R, SdkError> {
            let request = Request {
                id: self.mint_id(),
                method,
                params,
            };
            let response = self.transport.request(request).await?;

            if let Some(err) = response.error {
                return Err(SdkError::Rpc {
                    method: method.as_str().to_string(),
                    code: err.code,
                    message: err.message,
                });
            }
            let result = response.result.ok_or_else(|| SdkError::MissingResult {
                method: method.as_str().to_string(),
            })?;
            serde_json::from_value(result).map_err(|source| SdkError::Decode {
                method: method.as_str().to_string(),
                source,
            })
        }

        // -- typed helpers -----------------------------------------------------

        /// Start a session in a workspace (`session/new`). Returns the created
        /// [`Session`](crate::model::Session).
        pub async fn session_start<R: DeserializeOwned>(
            &self,
            workspace: &str,
            title: Option<&str>,
        ) -> Result<R, SdkError> {
            self.call(
                Method::SessionNew,
                serde_json::json!({ "workspace": workspace, "title": title }),
            )
            .await
        }

        /// Start a turn on a thread (`turn/create`). Returns the created
        /// [`Turn`](crate::model::Turn).
        pub async fn turn_start<R: DeserializeOwned>(
            &self,
            thread: &str,
            text: &str,
        ) -> Result<R, SdkError> {
            self.call(
                Method::TurnCreate,
                serde_json::json!({ "thread": thread, "text": text }),
            )
            .await
        }

        /// Fork a thread (`thread/fork`). Returns the new
        /// [`Thread`](crate::model::Thread).
        pub async fn thread_fork<R: DeserializeOwned>(&self, thread: &str) -> Result<R, SdkError> {
            self.call(Method::ThreadFork, serde_json::json!({ "thread": thread }))
                .await
        }

        /// Subscribe to a thread's item stream (`item/subscribe`). The subscribe
        /// call returns an ack (decoded and discarded); the items themselves arrive
        /// as [`Notification`]s, which this drains from the transport. A live
        /// streaming subscription is DEFERRED_MODEL_REQUIRED.
        pub async fn item_subscribe(&self, thread: &str) -> Result<Vec<Notification>, SdkError> {
            let _ack: Value = self
                .call(
                    Method::ItemSubscribe,
                    serde_json::json!({ "thread": thread }),
                )
                .await?;
            Ok(self.transport.notifications().await)
        }
    }

    /// A deterministic in-memory transport for tests. Register a canned result (or
    /// error) per method, preload notifications, and inspect the requests it
    /// received. No network, no model.
    #[derive(Default)]
    pub struct MockTransport {
        results: Mutex<HashMap<String, Value>>,
        errors: Mutex<HashMap<String, RpcError>>,
        notifications: Mutex<Vec<Notification>>,
        received: Mutex<Vec<Request>>,
    }

    impl MockTransport {
        /// A transport with no handlers. Add them with [`MockTransport::on`].
        pub fn new() -> Self {
            Self::default()
        }

        /// Register the canned result a method returns. Builder-style.
        pub fn on(self, method: Method, result: Value) -> Self {
            self.results
                .lock()
                .unwrap()
                .insert(method.as_str().to_string(), result);
            self
        }

        /// Register a protocol error a method returns instead of a result.
        pub fn on_error(self, method: Method, error: RpcError) -> Self {
            self.errors
                .lock()
                .unwrap()
                .insert(method.as_str().to_string(), error);
            self
        }

        /// Preload a notification the client will drain on the next
        /// [`Transport::notifications`] call.
        pub fn push_notification(self, notification: Notification) -> Self {
            self.notifications.lock().unwrap().push(notification);
            self
        }

        /// Every request the transport has received so far, in order.
        pub fn received(&self) -> Vec<Request> {
            self.received.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Transport for MockTransport {
        async fn request(&self, request: Request) -> Result<Response, SdkError> {
            let key = request.method.as_str().to_string();
            self.received.lock().unwrap().push(request.clone());

            if let Some(err) = self.errors.lock().unwrap().get(&key).cloned() {
                return Ok(Response {
                    id: request.id,
                    result: None,
                    error: Some(err),
                });
            }
            match self.results.lock().unwrap().get(&key).cloned() {
                Some(result) => Ok(Response {
                    id: request.id,
                    result: Some(result),
                    error: None,
                }),
                None => Err(SdkError::Unhandled(key)),
            }
        }

        async fn notifications(&self) -> Vec<Notification> {
            std::mem::take(&mut *self.notifications.lock().unwrap())
        }
    }
}

pub mod command {
    //! Project the ONE command registry from `hide-protocol` into the artifacts the
    //! frontend consumes: the serialized catalog (`command_catalog.json`) and the
    //! TypeScript that exposes the `CommandSpec` type plus the catalog array
    //! (`commands.d.ts`).
    //!
    //! Same guarantee as the protocol codegen (Bible sec 15.7): the FE does not
    //! hand-declare the command table. It comes from `crate::command_catalog`
    //! through this deterministic emitter, so a catalog change flows into the golden
    //! and fails the build until the FE artifact is regenerated.
    //!
    //! Model-free: pure deterministic codegen over a static table.

    use crate::{command_catalog, CommandSpec};

    /// The serialized command catalog, pretty-printed and newline-terminated. This
    /// is the exact `generated/command_catalog.json` artifact (data, not source LOC).
    pub fn command_catalog_json() -> String {
        let mut s = serde_json::to_string_pretty(&command_catalog())
            .expect("the command catalog is plain data and always serializes");
        s.push('\n');
        s
    }

    /// The command TypeScript: the `CommandSpec` interface (and every enum it
    /// references) generated from the schemars schema, plus a declare-only catalog
    /// binding. Checked in at `goldens/commands.d.ts` (counted active source).
    /// Runtime values live in `command_catalog.json` (loaded by the FE).
    pub fn command_typescript() -> String {
        let (_roots, defs) = crate::sdk::schema::collect_root::<CommandSpec>();

        let mut out = String::new();
        out.push_str("// Generated by hide-sdk from the hide-protocol command registry. DO NOT EDIT BY HAND.\n");
        out.push_str(
            "// The ONE command table: hide-protocol owns CommandSpec + command_catalog().\n",
        );
        out.push_str("// Regenerate with `cargo run -p hide-protocol --bin hide-sdk-codegen`.\n");
        out.push_str("// A drift from the Rust catalog is caught by the golden-file test.\n\n");

        out.push_str(&crate::sdk::ts::emit_definitions(&defs));

        // Runtime catalog values are command_catalog.json (FE loads that file).
        // Types-only here: the table is already the counted Rust authority.
        out.push_str("// Runtime values: import command_catalog.json (generated sibling).\n");
        out.push_str("export declare const COMMAND_CATALOG: CommandSpec[];\n");
        out
    }
}

pub mod fixtures {
    //! Canonical event fixtures.
    //!
    //! A small, deterministic set of [`Notification`] and [`Item`] values, built
    //! from `hide-protocol` types, that the compatibility tests round-trip through
    //! serde. Because the fixtures are typed hide-protocol values, they cannot
    //! encode a shape the protocol does not accept, and [`events_json`] renders them
    //! into the golden artifact the frontend and external clients can pin against.
    //!
    //! One source: the typed fixtures below are the origin; the JSON bundle and its
    //! golden are serialized from them, never maintained by hand.

    use serde_json::{Map, Value};

    use crate::ids::{ApprovalId, ItemId, PlanId, StepId, ToolCallId, ToolId, TurnId};
    use crate::item::{
        AgentMessage, ApprovalRequest, Completion, Item, ItemKind, ToolCall, ToolResult,
        UserMessage,
    };
    use crate::model::{CompletionStatus, Risk};
    use crate::plan::{Cost, Effect, Plan, PlanStep, RollbackBoundary, Scope};
    use crate::protocol::Notification;

    fn item(id: &str, seq: u64, kind: ItemKind) -> Item {
        Item {
            id: ItemId::from(id),
            turn: Some(TurnId::from("trn_1")),
            seq,
            kind,
            created_ms: 1_000 + seq,
        }
    }

    fn sample_plan() -> Plan {
        Plan {
            id: PlanId::from("pln_1"),
            goal: None,
            steps: vec![PlanStep {
                id: StepId::from("stp_1"),
                objective: "reproduce the flake".into(),
                dependencies: vec![],
                scope: Scope {
                    paths: vec!["src/retry.ts".into()],
                    network: false,
                    description: Some("the retry module only".into()),
                },
                effects: vec![Effect::ReadFs, Effect::Shell],
                expected_artifacts: vec!["repro.log".into()],
                acceptance_oracle: None,
                rollback_boundary: RollbackBoundary::default(),
                cost: Cost {
                    tokens: Some(1200),
                    wall_ms: Some(30_000),
                    usd_micros: None,
                },
                parallelizable: false,
            }],
            created_ms: 900,
        }
    }

    /// The canonical [`Item`] fixtures, each with a stable name.
    pub fn item_fixtures() -> Vec<(&'static str, Item)> {
        vec![
            (
                "user_message",
                item(
                    "itm_user",
                    0,
                    ItemKind::UserMessage(UserMessage {
                        text: "the retry test flakes on CI".into(),
                        attachments: vec![],
                    }),
                ),
            ),
            (
                "agent_message",
                item(
                    "itm_agent",
                    1,
                    ItemKind::AgentMessage(AgentMessage {
                        text: "reproducing it now".into(),
                    }),
                ),
            ),
            ("plan", item("itm_plan", 2, ItemKind::Plan(sample_plan()))),
            (
                "tool_call",
                item(
                    "itm_call",
                    3,
                    ItemKind::ToolCall(ToolCall {
                        call_id: ToolCallId::from("tcl_1"),
                        tool: ToolId::from("tool_bash"),
                        arguments: serde_json::json!({ "cmd": "cargo test -p retry" }),
                    }),
                ),
            ),
            (
                "tool_result",
                item(
                    "itm_result",
                    4,
                    ItemKind::ToolResult(ToolResult {
                        call_id: ToolCallId::from("tcl_1"),
                        ok: true,
                        output: serde_json::json!({ "code": 0, "passed": 12 }),
                        error: None,
                    }),
                ),
            ),
            (
                "completion",
                item(
                    "itm_done",
                    5,
                    ItemKind::Completion(Completion {
                        status: CompletionStatus::Success,
                        summary: Some("retry test is green".into()),
                    }),
                ),
            ),
        ]
    }

    /// The canonical [`Notification`] fixtures, each with a stable name.
    pub fn notification_fixtures() -> Vec<(&'static str, Notification)> {
        vec![
            (
                "turn_started",
                Notification::TurnStarted {
                    turn: TurnId::from("trn_1"),
                },
            ),
            (
                "item_added",
                Notification::ItemAdded {
                    item: item(
                        "itm_agent",
                        1,
                        ItemKind::AgentMessage(AgentMessage {
                            text: "reproducing it now".into(),
                        }),
                    ),
                },
            ),
            (
                "approval_requested",
                Notification::ApprovalRequested {
                    request: ApprovalRequest {
                        request_id: ApprovalId::from("apr_1"),
                        action: "write src/retry.ts".into(),
                        risk: Risk::Low,
                        effects: vec![Effect::WriteFs],
                        detail: Some("apply the one-line fix".into()),
                    },
                },
            ),
            (
                "runtime_status",
                Notification::RuntimeStatus {
                    status: "ready".into(),
                    detail: None,
                },
            ),
        ]
    }

    /// The full event bundle as a [`serde_json::Value`]:
    ///
    /// ```json
    /// { "items": { "<name>": { ... } }, "notifications": { "<name>": { ... } } }
    /// ```
    ///
    /// Serialized from the typed fixtures, so it is a faithful projection of the
    /// protocol shapes, not a hand-written mirror.
    pub fn events_bundle() -> Value {
        let mut items = Map::new();
        for (name, value) in item_fixtures() {
            items.insert(
                name.to_string(),
                serde_json::to_value(&value).expect("an Item always serializes"),
            );
        }
        let mut notifications = Map::new();
        for (name, value) in notification_fixtures() {
            notifications.insert(
                name.to_string(),
                serde_json::to_value(&value).expect("a Notification always serializes"),
            );
        }

        let mut bundle = Map::new();
        bundle.insert("items".to_string(), Value::Object(items));
        bundle.insert("notifications".to_string(), Value::Object(notifications));
        Value::Object(bundle)
    }

    /// The event bundle as a stable, pretty-printed string: the golden fixtures
    /// artifact.
    pub fn events_json() -> String {
        let mut s = serde_json::to_string_pretty(&events_bundle())
            .expect("the bundle is plain JSON and always serializes");
        s.push('\n');
        s
    }
}

pub mod schema {
    //! Schema export: emit the protocol JSON Schema from the ONE source.
    //!
    //! hide-protocol derives schemars on every wire type. This module collects the
    //! top protocol types into a single, deterministic JSON Schema bundle (an
    //! OpenAPI-`components`-style document with a shared `definitions` map), so the
    //! whole protocol is one artifact that codegen, contract tests, and a published
    //! schema bundle all render from. Because the schemas come from the Rust types,
    //! the bundle can never silently drift from the code.
    //!
    //! Determinism: schemars' `definitions` and each object's `properties` /
    //! `required` are backed by `BTreeMap`/`BTreeSet`, and serde_json here is built
    //! without `preserve_order`, so the serialized bundle is byte-stable run to run.
    //! That stability is what makes the golden-file test meaningful.

    use schemars::gen::SchemaGenerator;
    use schemars::JsonSchema;
    use serde_json::{Map, Value};

    use crate::{
        Agent, InitializeRequest, InitializeResult, Item, Method, Notification, Plan, Session,
        Thread, Turn, PROTOCOL_VERSION,
    };

    /// The top protocol types the bundle roots at, in a fixed order. These are the
    /// entry points a frontend or an external client cares about; every other type
    /// they reference is pulled into `definitions` transitively.
    ///
    /// "Initialize" from the Bible list expands to both halves of the handshake:
    /// [`InitializeRequest`] and [`InitializeResult`].
    pub const ROOT_TYPE_NAMES: &[&str] = &[
        "Session",
        "Thread",
        "Turn",
        "Item",
        "Method",
        "Notification",
        "InitializeRequest",
        "InitializeResult",
        "Plan",
        "Agent",
    ];

    /// Register one root type into the generator and record its schema name.
    fn register<T: JsonSchema>(gen: &mut SchemaGenerator, roots: &mut Vec<String>) {
        // subschema_for adds T (and everything it references) to the generator's
        // definitions and returns a $ref we do not need to keep here.
        let _ = gen.subschema_for::<T>();
        roots.push(T::schema_name());
    }

    /// Collect every root type's schema into one shared generator and return its
    /// definitions map (type name -> JSON Schema), plus the ordered root names.
    ///
    /// The definitions map is a `BTreeMap`, so iteration and serialization are
    /// deterministic.
    pub fn collect_definitions() -> (Vec<String>, Map<String, Value>) {
        let mut gen = SchemaGenerator::default();
        let mut roots = Vec::new();

        register::<Session>(&mut gen, &mut roots);
        register::<Thread>(&mut gen, &mut roots);
        register::<Turn>(&mut gen, &mut roots);
        register::<Item>(&mut gen, &mut roots);
        register::<Method>(&mut gen, &mut roots);
        register::<Notification>(&mut gen, &mut roots);
        register::<InitializeRequest>(&mut gen, &mut roots);
        register::<InitializeResult>(&mut gen, &mut roots);
        register::<Plan>(&mut gen, &mut roots);
        register::<Agent>(&mut gen, &mut roots);

        let mut definitions = Map::new();
        for (name, schema) in gen.definitions().iter() {
            let value = serde_json::to_value(schema)
                .expect("a schemars Schema always serializes to serde_json::Value");
            definitions.insert(name.clone(), value);
        }

        (roots, definitions)
    }

    /// Collect a single root type's schema (plus everything it references) into a
    /// deterministic definitions map, for a codegen surface that roots on one type
    /// (the command catalog roots on [`CommandSpec`](crate::CommandSpec)).
    pub fn collect_root<T: JsonSchema>() -> (Vec<String>, Map<String, Value>) {
        let mut gen = SchemaGenerator::default();
        let mut roots = Vec::new();
        register::<T>(&mut gen, &mut roots);

        let mut definitions = Map::new();
        for (name, schema) in gen.definitions().iter() {
            let value = serde_json::to_value(schema)
                .expect("a schemars Schema always serializes to serde_json::Value");
            definitions.insert(name.clone(), value);
        }
        (roots, definitions)
    }

    /// The full protocol JSON Schema bundle as a [`serde_json::Value`].
    ///
    /// Shape:
    ///
    /// ```json
    /// {
    ///   "$schema": "http://json-schema.org/draft-07/schema#",
    ///   "title": "HIDE Agent Protocol",
    ///   "protocolVersion": "hide.agent.v1",
    ///   "roots": ["Session", "Thread", ...],
    ///   "definitions": { "Session": { ... }, ... }
    /// }
    /// ```
    pub fn protocol_schema_bundle() -> Value {
        let (roots, definitions) = collect_definitions();

        let mut bundle = Map::new();
        bundle.insert(
            "$schema".to_string(),
            Value::String("http://json-schema.org/draft-07/schema#".to_string()),
        );
        bundle.insert(
            "title".to_string(),
            Value::String("HIDE Agent Protocol".to_string()),
        );
        bundle.insert(
            "protocolVersion".to_string(),
            Value::String(PROTOCOL_VERSION.to_string()),
        );
        bundle.insert(
            "roots".to_string(),
            Value::Array(roots.into_iter().map(Value::String).collect()),
        );
        bundle.insert("definitions".to_string(), Value::Object(definitions));

        Value::Object(bundle)
    }

    /// The protocol JSON Schema bundle as a stable, pretty-printed string. This is
    /// the exact artifact the golden-file test pins.
    pub fn protocol_schema_json() -> String {
        let mut s = serde_json::to_string_pretty(&protocol_schema_bundle())
            .expect("the bundle is plain JSON and always serializes");
        s.push('\n');
        s
    }
}

pub mod ts {
    //! Deterministic JSON-Schema-to-TypeScript emitter.
    //!
    //! This is the "no handwritten frontend mirror types" guarantee (Bible sec
    //! 15.7): the frontend's protocol types are generated here, from the same
    //! schemars schemas [`crate::sdk::schema`] exports, so they cannot drift from the
    //! Rust wire types. A protocol change reshapes this output, and the golden-file
    //! test then fails until the frontend types are regenerated.
    //!
    //! # Covered subset
    //!
    //! The emitter maps the JSON Schema shapes hide-protocol actually produces:
    //!
    //! - objects (`type: "object"` with `properties`) -> `export interface`, with a
    //!   field made optional (`?`) when it is absent from `required`;
    //! - string enums (`type: "string"` with `enum`) -> `export type` string-literal
    //!   unions (a single-value enum becomes one string literal, which is how the
    //!   `kind` / `method` discriminants render);
    //! - tagged unions (`oneOf` / `anyOf`) -> `export type` unions, including the
    //!   `Option<Ref>` shape `anyOf: [Ref, null]` -> `Ref | null`;
    //! - nullable primitives (`type: ["string", "null"]`) -> `string | null`;
    //! - arrays (`type: "array"`) -> `T[]` (unions parenthesized: `(A | B)[]`);
    //! - `$ref` -> the referenced type name;
    //! - maps (`type: "object"` with `additionalProperties`) -> `Record<string, V>`;
    //! - the permissive schemas `true` / `{}` (schemars' rendering of
    //!   `serde_json::Value`) -> `unknown`.
    //!
    //! Anything outside that subset falls back to `unknown` rather than emitting a
    //! wrong type. hide-protocol stays inside the subset today; a future shape that
    //! needs more (for example numeric enums or `allOf` intersections) would show up
    //! as `unknown` in the golden and is the documented deferred remainder.
    //!
    //! # No dashes by construction
    //!
    //! Descriptions carried from Rust doc comments are normalized to ASCII hyphens
    //! ([`normalize`]) before they enter a JSDoc block, so the generated `.ts`
    //! stays free of en/em dashes regardless of upstream punctuation.

    use serde_json::{Map, Value};

    use crate::sdk::schema::collect_definitions;

    /// Emit the full protocol TypeScript declarations as a stable string. This is
    /// the exact artifact the golden-file test pins and the frontend consumes.
    pub fn protocol_typescript() -> String {
        let (roots, defs) = collect_definitions();
        emit(&roots, &defs)
    }

    /// Emit `export` declarations for a set of schemars definitions, without the
    /// protocol file header. Used by codegen surfaces that supply their own header
    /// and then append their own data (the command catalog appends its const array).
    pub fn emit_definitions(defs: &Map<String, Value>) -> String {
        let mut out = String::new();
        for (name, schema) in defs.iter() {
            emit_declaration(name, schema, &mut out);
        }
        out
    }

    fn emit(roots: &[String], defs: &Map<String, Value>) -> String {
        let mut out = String::new();
        out.push_str("// Generated by hide-sdk from hide-protocol. DO NOT EDIT BY HAND.\n");
        out.push_str("// Bible sec 15.7: one schema source generates the frontend types.\n");
        out.push_str("// Regenerate with `cargo run -p hide-sdk --bin hide-sdk-codegen`.\n");
        out.push_str("// A drift from the Rust wire types is caught by the golden-file test.\n");
        out.push_str("// Protocol version: hide.agent.v1\n");
        out.push_str(&format!("// Root types: {}\n\n", roots.join(", ")));

        // defs is a serde_json object (BTreeMap-backed, no preserve_order), so this
        // iteration is alphabetical and deterministic.
        for (name, schema) in defs.iter() {
            emit_declaration(name, schema, &mut out);
        }
        // Single trailing newline only (git diff --check rejects blank line at EOF).
        while out.ends_with("\n\n") {
            out.pop();
        }
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out
    }

    fn emit_declaration(name: &str, schema: &Value, out: &mut String) {
        if let Some(desc) = schema.get("description").and_then(Value::as_str) {
            push_doc(out, desc, "");
        }

        // String enum -> string-literal union alias.
        if let Some(en) = schema.get("enum").and_then(Value::as_array) {
            out.push_str(&format!(
                "export type {name} = {};\n\n",
                union(literals(en))
            ));
            return;
        }

        // Tagged union -> union alias.
        if let Some(variants) = one_of(schema) {
            let parts = variants.iter().map(ts_type).collect::<Vec<_>>();
            out.push_str(&format!("export type {name} = {};\n\n", union(parts)));
            return;
        }

        if schema.get("type").and_then(Value::as_str) == Some("object") {
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                let required = required_set(schema);
                out.push_str(&format!("export interface {name} {{\n"));
                for (field, field_schema) in props.iter() {
                    if let Some(desc) = field_schema.get("description").and_then(Value::as_str) {
                        push_doc(out, desc, "  ");
                    }
                    let opt = if required.contains(field.as_str()) {
                        ""
                    } else {
                        "?"
                    };
                    out.push_str(&format!(
                        "  {}{}: {};\n",
                        ident(field),
                        opt,
                        ts_type(field_schema)
                    ));
                }
                out.push_str("}\n\n");
                return;
            }
            // A map (additionalProperties) with no fixed properties.
            out.push_str(&format!("export type {name} = {};\n\n", ts_type(schema)));
            return;
        }

        // Anything else: a plain alias to the mapped type.
        out.push_str(&format!("export type {name} = {};\n\n", ts_type(schema)));
    }

    /// Map one schema node to a TypeScript type expression.
    fn ts_type(schema: &Value) -> String {
        let map = match schema {
            // schemars renders `serde_json::Value` as the permissive boolean schema
            // `true`; `false` is the empty/never schema. Both become `unknown`.
            Value::Bool(_) => return "unknown".to_string(),
            Value::Object(map) => map,
            _ => return "unknown".to_string(),
        };

        if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
            return ref_name(reference);
        }
        if let Some(en) = map.get("enum").and_then(Value::as_array) {
            return union(literals(en));
        }
        if let Some(variants) = one_of(schema) {
            let parts = variants.iter().map(ts_type).collect::<Vec<_>>();
            return union(parts);
        }

        match map.get("type") {
            Some(Value::String(t)) => ts_primitive(t, map),
            Some(Value::Array(types)) => {
                let parts = types
                    .iter()
                    .map(|t| ts_primitive(t.as_str().unwrap_or("unknown"), map))
                    .collect::<Vec<_>>();
                union(parts)
            }
            _ => "unknown".to_string(),
        }
    }

    fn ts_primitive(t: &str, map: &Map<String, Value>) -> String {
        match t {
            "string" => "string".to_string(),
            "integer" | "number" => "number".to_string(),
            "boolean" => "boolean".to_string(),
            "null" => "null".to_string(),
            "array" => {
                let inner = map
                    .get("items")
                    .map(ts_type)
                    .unwrap_or_else(|| "unknown".to_string());
                format!("{}[]", parenthesize(&inner))
            }
            "object" => {
                if let Some(props) = map.get("properties").and_then(Value::as_object) {
                    inline_object(props, map)
                } else {
                    match map.get("additionalProperties") {
                        Some(Value::Object(_)) => {
                            let value = ts_type(map.get("additionalProperties").unwrap());
                            format!("Record<string, {value}>")
                        }
                        Some(Value::Bool(false)) => "Record<string, never>".to_string(),
                        _ => "Record<string, unknown>".to_string(),
                    }
                }
            }
            _ => "unknown".to_string(),
        }
    }

    /// An inline object literal type, used by `oneOf` variants (Item, Notification)
    /// and inline `params` objects.
    fn inline_object(props: &Map<String, Value>, schema: &Map<String, Value>) -> String {
        let required = required_set(&Value::Object(schema.clone()));
        let mut s = String::from("{ ");
        for (field, field_schema) in props.iter() {
            let opt = if required.contains(field.as_str()) {
                ""
            } else {
                "?"
            };
            s.push_str(&format!(
                "{}{}: {}; ",
                ident(field),
                opt,
                ts_type(field_schema)
            ));
        }
        s.push('}');
        s
    }

    // -- helpers ---------------------------------------------------------------

    fn one_of(schema: &Value) -> Option<&Vec<Value>> {
        schema
            .get("oneOf")
            .or_else(|| schema.get("anyOf"))
            .and_then(Value::as_array)
    }

    fn literals(values: &[Value]) -> Vec<String> {
        values
            .iter()
            .map(|v| match v {
                Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
                other => other.to_string(),
            })
            .collect()
    }

    fn union(parts: Vec<String>) -> String {
        if parts.is_empty() {
            return "never".to_string();
        }
        parts.join(" | ")
    }

    /// Wrap a union in parentheses so `A | B` arrays render as `(A | B)[]`.
    fn parenthesize(ty: &str) -> String {
        if ty.contains(" | ") {
            format!("({ty})")
        } else {
            ty.to_string()
        }
    }

    fn required_set(schema: &Value) -> std::collections::BTreeSet<String> {
        schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn ref_name(reference: &str) -> String {
        reference
            .rsplit('/')
            .next()
            .unwrap_or(reference)
            .to_string()
    }

    /// A field name is used bare when it is a valid TS identifier, else quoted.
    fn ident(field: &str) -> String {
        let valid = !field.is_empty()
            && field.chars().enumerate().all(|(i, c)| {
                c == '_' || c == '$' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
            });
        if valid {
            field.to_string()
        } else {
            format!("\"{}\"", field.replace('"', "\\\""))
        }
    }

    /// Emit a JSDoc block at the given indent, one line per source line, with
    /// dashes normalized so the output is ASCII-clean.
    fn push_doc(out: &mut String, desc: &str, indent: &str) {
        out.push_str(&format!("{indent}/**\n"));
        for line in normalize(desc).split('\n') {
            let line = line.replace("*/", "* /");
            // No trailing space on empty JSDoc lines (git diff --check).
            if line.is_empty() {
                out.push_str(&format!("{indent} *\n"));
            } else {
                out.push_str(&format!("{indent} * {line}\n"));
            }
        }
        out.push_str(&format!("{indent} */\n"));
    }

    /// Replace en dash (U+2013) and em dash (U+2014) with a plain hyphen so no
    /// generated text can carry a non-hyphen dash. Idempotent on ASCII input.
    pub fn normalize(text: &str) -> String {
        text.replace('\u{2013}', "-").replace('\u{2014}', "-")
    }
}

pub use client::{Client, MockTransport, SdkError, Transport};
pub use command::{command_catalog_json, command_typescript};
pub use schema::{protocol_schema_bundle, protocol_schema_json, ROOT_TYPE_NAMES};
pub use ts::protocol_typescript;
