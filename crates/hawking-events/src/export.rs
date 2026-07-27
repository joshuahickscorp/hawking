//! Export `HAWKING_CANONICAL_EVENTS.json` (schema `hawking.events.canonical.v1`).

use serde_json::{json, Value};

use crate::categories::{kind_for_category, Category, CATEGORY_KINDS};
use crate::envelope::CANONICAL_SCHEMA;
use crate::models::{MigrationStatus, COMPETING_MODELS};
use crate::DOCUMENT_SCHEMA;

/// Build the deliverable document as a JSON Value.
pub fn canonical_events_document() -> Value {
    let categories: Vec<Value> = CATEGORY_KINDS
        .iter()
        .map(|(cat, kind)| {
            json!({
                "category": cat.as_str(),
                "primary_kind": kind,
                "covered": true,
            })
        })
        .collect();

    let superseded: Vec<Value> = COMPETING_MODELS
        .iter()
        .filter(|m| m.status != MigrationStatus::Canonical)
        .map(|m| {
            json!({
                "name": m.name,
                "file": m.file,
                "line": m.line,
                "role": m.role,
                "status": match m.status {
                    MigrationStatus::Canonical => "canonical",
                    MigrationStatus::LiveProjection => "live_projection",
                    MigrationStatus::DeprecatedAdapted => "deprecated_adapted",
                    MigrationStatus::CampaignLedger => "campaign_ledger",
                },
                "adapter_module": m.adapter_module,
                "notes": m.notes,
            })
        })
        .collect();

    let chosen = COMPETING_MODELS
        .iter()
        .find(|m| m.status == MigrationStatus::Canonical)
        .expect("exactly one canonical model");

    json!({
        "schema": DOCUMENT_SCHEMA,
        "canonical_schema": CANONICAL_SCHEMA,
        "chosen_model": {
            "name": chosen.name,
            "file": chosen.file,
            "line": chosen.line,
            "justification": "Among models that can express the full category surface (model lifecycle, text, reasoning, plans, tools, permissions, edits, tests, verification, agents, acceleration, Fabric placement/node state, warnings, errors, usage), hide-core Event already carries the durable product-event traffic: hide-backend appends it to JsonlEventLog and replays it into UiEvent. StreamEvent has higher token volume but only Token|Done and cannot be the product authority. Chosen for live durable traffic, not aesthetics.",
            "envelope_fields": ["id", "seq", "session_id", "subsystem", "verification"],
        },
        "superseded_models": superseded,
        "categories": categories,
        "category_count": Category::all().len(),
        "migration_status": {
            "hide_core_event": "canonical",
            "ui_event": "live_projection_with_adapter",
            "protocol_item": "live_projection_with_adapter",
            "stream_event": "live_projection_with_adapter",
            "seed_c_event": "deprecated_adapted",
            "campaign_jsonl_ledgers": "campaign_ledger_no_adapter",
        },
        "two_live_projections_loudly": [
            "StreamEvent remains the inference hot path (crates/hawking-core/src/engine.rs:188); project via hawking_events::adapters::stream.",
            "UiEvent remains Wire-B UI transport (crates/hide-core/src/api.rs:78); durable authority is still Event."
        ],
        "primary_kinds": CATEGORY_KINDS.iter().map(|(c, k)| json!({
            "category": c.as_str(),
            "kind": k,
        })).collect::<Vec<_>>(),
    })
}

/// Pretty-printed JSON for the checked-in deliverable / drift test.
pub fn canonical_events_json() -> String {
    let mut s = serde_json::to_string_pretty(&canonical_events_document())
        .expect("document always serializes");
    s.push('\n');
    s
}

/// Resolve the primary kind for a category (export helper).
pub fn export_kind(cat: Category) -> &'static str {
    kind_for_category(cat)
}
