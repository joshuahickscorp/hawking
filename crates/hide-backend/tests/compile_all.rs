//! Compile-time coverage for every hide-backend integration source.
//! Live model and process tests are included for type checking only.

#[path = "background_and_lifecycle_trace_g.rs"]
mod background_and_lifecycle_trace_g;
#[path = "compat_compat.rs"]
mod compat_compat;
#[path = "connector_abi_safety_and_impls.rs"]
mod connector_abi_safety_and_impls;
#[path = "diff_review_trace_c.rs"]
mod diff_review_trace_c;
#[path = "first_model_free_receipt.rs"]
mod first_model_free_receipt;
#[path = "host_slice_core.rs"]
mod host_slice_core;
#[path = "lenses_properties.rs"]
mod lenses_properties;
#[path = "live_model_turn.rs"]
mod live_model_turn;
#[path = "os_core_blackbox.rs"]
mod os_core_blackbox;
#[path = "search_provenance_trace_f.rs"]
mod search_provenance_trace_f;
#[path = "steer_and_dispatch.rs"]
mod steer_and_dispatch;
#[path = "wave3_reachability.rs"]
mod wave3_reachability;
#[path = "wire_write_path.rs"]
mod wire_write_path;
