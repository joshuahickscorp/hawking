//! Enumeration of competing event models (archaeology verification).
//!
//! Each entry names a `file:line` anchor so the survey claim is re-checkable.

use serde::{Deserialize, Serialize};

/// Status of a non-canonical model relative to the hide-core Event authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationStatus {
    /// This *is* the canonical durable model.
    Canonical,
    /// Still live as a projection/hot-path; adapter into canonical exists.
    LiveProjection,
    /// Left in place with deprecation docs + adapter; do not expand.
    DeprecatedAdapted,
    /// Campaign/scientific ledger only; not a product event stream.
    CampaignLedger,
}

/// One competing event-shaped model found in the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetingModel {
    pub name: &'static str,
    pub file: &'static str,
    pub line: u32,
    pub role: &'static str,
    pub status: MigrationStatus,
    pub adapter_module: Option<&'static str>,
    pub notes: &'static str,
}

/// Verified against the worktree on 2026-07-26. Line numbers are the enum /
/// struct definition anchors.
pub const COMPETING_MODELS: &[CompetingModel] = &[
    CompetingModel {
        name: "hide_core::event::Event",
        file: "crates/hide-core/src/event.rs",
        line: 52,
        role: "Durable append-only product event log (id, seq, session, open kind)",
        status: MigrationStatus::Canonical,
        adapter_module: None,
        notes: "Chosen canonical. hide-backend JsonlEventLog + replay write/read this shape.",
    },
    CompetingModel {
        name: "hide_core::api::UiEvent",
        file: "crates/hide-core/src/api.rs",
        line: 78,
        role: "Wire-B UI bus (seq + session + UiEventKind)",
        status: MigrationStatus::LiveProjection,
        adapter_module: Some("hawking_events::adapters::ui"),
        notes: "Still live transport; hide-backend maps Event -> UiEvent. Not a second durable log.",
    },
    CompetingModel {
        name: "hide_protocol::item::Item / ItemKind",
        file: "crates/hide-protocol/src/item.rs",
        line: 287,
        role: "Turn wire schema authority (Items inside Turns)",
        status: MigrationStatus::LiveProjection,
        adapter_module: Some("hawking_events::adapters::item"),
        notes: "Schema authority for turn items remains; adapts into canonical kinds for the durable log.",
    },
    CompetingModel {
        name: "hawking_core::engine::StreamEvent",
        file: "crates/hawking-core/src/engine.rs",
        line: 188,
        role: "Inference hot-path token stream (Token | Done only)",
        status: MigrationStatus::LiveProjection,
        adapter_module: Some("hawking_events::adapters::stream"),
        notes: "Highest *token* traffic; too narrow for full categories. Projects to model.token/model.usage.",
    },
    CompetingModel {
        name: "hawking_seed_c::state::Event",
        file: "crates/hawking-events/src/adapters/seed.rs",
        line: 1,
        role: "Historical campaign FSM (binary product-released under BC-BRIDGE-012 / B-RT5)",
        status: MigrationStatus::DeprecatedAdapted,
        adapter_module: Some("hawking_events::adapters::seed"),
        notes: "hawking-seed-c crate deleted under B-RT5; hermetic mirror retained for seed.transition projection only. Do not expand as a product event bus.",
    },
    CompetingModel {
        name: "campaign JSONL ledgers",
        file: "evidence/hawking/HAWKING_CAMPAIGN_LEDGER.jsonl",
        line: 1,
        role: "Scientific/receipt campaign ledgers at repo root",
        status: MigrationStatus::CampaignLedger,
        adapter_module: None,
        notes: "Not product events; stay as campaign receipts. No adapter required.",
    },
];
