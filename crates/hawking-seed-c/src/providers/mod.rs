//! # Absorbed hawking-packs nucleus — capability providers over the one Pack ABI.
//!
//! Every active pack capability is expressed here as a **pure provider** orbiting this Seed's own
//! authority — it never recreates it. Providers reuse [`crate::record`] (canonical seal/identity),
//! [`crate::evidence`] (receipts), [`crate::state`] (the one controller), [`crate::pack`] (the one Pack
//! ABI + content verification), [`crate::gravity`] (the law), [`crate::ir`] (the execution IR), and
//! [`crate::subbit`] (the reference Forge/Doctor mathematics). This is the accretion disk; the Seed is the
//! black hole.
//!
//! - **one Pack ABI** ([`crate::pack::PackManifest`]) — one manifest schema, content-addressed;
//! - **one verifier** ([`verify`]) — offline hydration, tamper refusal, compatibility, rollback, active-set;
//! - **one capability registry** ([`registry`]) — which pack provides a capability, which impl is active,
//!   why, what ABI, LOC/bytes, source commit, tests, rollback;
//! - **shared provider traits** ([`provider`]) — a provider receives Seed-owned context and returns
//!   `{result, metrics, evidence, resource_usage}`; it may not mutate Seed state;
//! - **declarative adapters** ([`adapters`]), **one Forge contract** ([`forge`]), **one Doctor contract**
//!   ([`doctor`]), a **Metal op provider** ([`metal`]), a **speculation provider** ([`speculation`]),
//!   **one validation manifest** ([`validation`]), **one experiment schema** ([`experiment`]), **one
//!   source record** ([`source_decl`]), and **profile accounting** ([`profiles`]).

pub mod provider;
pub mod source_decl;
pub mod registry;
pub mod verify;
pub mod adapters;
pub mod forge;
pub mod doctor;
pub mod metal;
pub mod speculation;
pub mod validation;
pub mod experiment;
pub mod profiles;
