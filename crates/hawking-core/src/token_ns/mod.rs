//! Unified TOKEN_NS schema, adapters, closure identity, and lane reconciler.
//!
//! This module does not sit on a runtime hot path. Existing Q80 and DSV4F
//! collectors keep their own shapes; call [`from_dsv4f_ledger`] /
//! [`from_q80_ledger`] (or the JSON adapters) to emit the common document.

mod adapt;
mod audit;
pub mod energy;
mod reconcile;
mod schema;
pub mod served_weight;

pub use adapt::{
    from_dsv4f_json, from_dsv4f_ledger, from_q80_baseline_run_json, from_q80_json, from_q80_ledger,
    from_qwen38_ledger,
};
pub use audit::{flag_receipt, FlagSeverity, ReceiptFlag};
pub use energy::{
    probe_energy_model, EnergyProbeReport, EnergyReport, EnergySampler, EnergyScope,
    ENERGY_FILL_COMMAND, ENERGY_FILL_HOWTO,
};
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
pub use served_weight::{
    dsv4f_geometry, q80_geometry, qwen38_geometry, ActiveWeightGeometry, ModelId, ServedWeightHonesty,
    ServedWeightMetrics, AMORTIZED_CAVEAT, FS_PER_WEIGHT_SERVED_FIELD, HARDWARE_PS_PER_BIT,
    M3_ULTRA_96GB_PEAK_BYTES_PER_S,
};
