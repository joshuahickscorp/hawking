//! Benchmark suites. Each one is independently runnable via
//! `dismantle bench --suite <name>`. The `competitive` suite is the
//! head-to-head comparison against llama.cpp. (Was `wax` pre-2026-04
//! audit; the alias still works.)

pub mod bandwidth;
pub mod competitive;
pub mod decode;
pub mod prefill;
pub mod throughput;
