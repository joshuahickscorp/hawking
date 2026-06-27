//! HIDE model orchestration.
//!
//! The orchestrator is a shell-side router over one or more `hawking serve`
//! instances or other providers. It is HTTP/interface only: no GPU code here.

pub mod difficulty;
pub mod grammar;
pub mod http_client;
pub mod inference;
pub mod registry;
pub mod router;
pub mod sampler;
pub mod supervisor;

pub use registry::{default_hawking_local_roles, RoleRegistry};
pub use router::{RouteDecision, Router, SimpleRouter};
