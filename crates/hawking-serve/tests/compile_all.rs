//! Compile-time coverage for every hawking-serve integration source.
//! The fast lane selects this normal test target with `cargo check`.

#[path = "effective_policy.rs"]
mod effective_policy;
#[path = "energy_gather_window.rs"]
mod energy_gather_window;
#[path = "greedy_lane_routing.rs"]
mod greedy_lane_routing;
#[path = "hawking_native_endpoint.rs"]
mod hawking_native_endpoint;
#[path = "http_integration.rs"]
mod http_integration;
#[path = "system_kv_bank.rs"]
mod system_kv_bank;
#[path = "system_kv_bank_wiring.rs"]
mod system_kv_bank_wiring;
#[path = "workload_pack_mapping.rs"]
mod workload_pack_mapping;
