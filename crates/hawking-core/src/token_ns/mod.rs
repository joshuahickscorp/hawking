//! Unified TOKEN_NS schema, adapters, closure identity, and lane reconciler.
//!
//! This module does not sit on a runtime hot path. Existing Q80 and DSV4F
//! collectors keep their own shapes; call [`from_dsv4f_ledger`] /
//! [`from_q80_ledger`] (or the JSON adapters) to emit the common document.

mod adapt;
mod audit;
mod reconcile;
mod schema;

pub use adapt::{
    from_dsv4f_json, from_dsv4f_ledger, from_q80_baseline_run_json, from_q80_json, from_q80_ledger,
};
pub use audit::{flag_receipt, FlagSeverity, ReceiptFlag};
pub use reconcile::{
    ascent_2026_08_16_claims, ascent_2026_08_16_kinds, ascent_2026_08_16_measured,
    ascent_2026_08_16_reports, reconcile, with_kinds, DiscrepancyKind, LaneClaim, LaneReconciliation,
    MeasuredToken, ModelReconciliation,
};
pub use schema::{
    ClosureReport, Confidence, CriticalPath, EmitMeta, MeasurementLabel, RemovableOrNecessary,
    ResourceClass, SerialOrOverlappable, TokenNsDocument, TokenNsStage, TokenNsTotals,
    DEFAULT_RESIDUAL_LIMIT, GPU_TIMESTAMP_AUTHORITY, TOKEN_NS_SCHEMA,
};
