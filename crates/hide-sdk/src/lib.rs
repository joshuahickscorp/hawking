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
//!   `hide-protocol` [`Method`](hide_protocol::Method) requests and parse typed
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

pub mod client;
pub mod command;
pub mod fixtures;
pub mod schema;
pub mod ts;

pub use client::{Client, MockTransport, SdkError, Transport};
pub use command::{command_catalog_json, command_typescript};
pub use schema::{protocol_schema_bundle, protocol_schema_json, ROOT_TYPE_NAMES};
pub use ts::protocol_typescript;
