//! # Canonical Hawking event model
//!
//! **Authority:** [`hide_core::event::Event`] is the single durable event
//! record. Every other event-shaped type in the workspace is either a
//! projection of this model or a **deprecated** peer that adapts *into* it.
//!
//! ## Why this one (traffic, not aesthetics)
//!
//! Six competing models were enumerated (see [`models`]):
//!
//! | Model | Location | Live role |
//! |---|---|---|
//! | **hide-core `Event`** | `hide-core/src/event.rs` | Durable log + replay in hide-backend |
//! | `UiEvent` | `hide-core/src/api.rs` | Wire-B UI bus (projection of Event) |
//! | hide-protocol `Item` | `hide-protocol/src/item.rs` | Turn wire schema (schema authority for items) |
//! | `StreamEvent` | `hawking-core/src/engine.rs` | Hot-path token stream (Token/Done only) |
//! | seed-c `state::Event` (historical) | released under BC-BRIDGE-012; hermetic mirror in `adapters::seed` | Former campaign FSM only |
//! | Campaign JSONL ledgers | repo-root `*_LEDGER.jsonl` | Scientific/receipt logs, not product events |
//!
//! `StreamEvent` carries the most *token* traffic but is too narrow to cover
//! plans/tools/agents/Fabric. Among models that can express the full category
//! surface, **hide-core `Event`** is already the durable append-only log that
//! hide-backend writes and replays (`event_to_ui_event`, JsonlEventLog). That
//! is the most live *product-event* traffic, so it is canonical.
//!
//! ## Envelope contract
//!
//! Every canonical event carries:
//! - stable id (`Event.id`)
//! - monotone sequence (`Event.seq`)
//! - session identity (`Event.session_id`)
//! - producing surface (`CanonicalEvent.surface`)
//! - producing subsystem (`CanonicalEvent.subsystem` / `Event.actor`)
//! - verification status (`CanonicalEvent.verification`) — target-verified or provisional
//!
//! ## YOU events
//!
//! Seventeen YOU surface events ([`you_events`]) join **this same bus** as
//! open kinds (`you.object.added`, …). They are not a second event bus.
//!
//! ## Two live projections (loudly documented)
//!
//! - **`StreamEvent`** remains the inference hot path. Adapters project it into
//!   `model.token` / `model.usage` kinds; it is *not* a second durable authority.
//! - **`UiEvent`** remains Wire-B UI transport. It is a projection of the durable
//!   log (see hide-backend `replay::event_to_ui_event`).
//!
//! Do not introduce a third durable log.

pub mod adapters;
pub mod categories;
pub mod envelope;
pub mod export;
pub mod models;
pub mod you_events;

pub use categories::{
    all_categories, category_for_kind, kind_for_category, Category, CATEGORY_KINDS,
};
pub use envelope::{
    stamp_legacy, stamp_legacy_with_surface, CanonicalEvent, ContentVerification, NewCanonical,
    ProducingSurface, Subsystem, CANONICAL_SCHEMA,
};
pub use export::{canonical_events_document, canonical_events_json};
pub use models::{CompetingModel, MigrationStatus, COMPETING_MODELS};
pub use you_events::{you_events_export, YouEvent, YouEventSpec, YOU_EVENTS};

/// Schema id for the checked-in `HAWKING_CANONICAL_EVENTS.json` deliverable.
pub const DOCUMENT_SCHEMA: &str = "hawking.events.canonical.v1";
