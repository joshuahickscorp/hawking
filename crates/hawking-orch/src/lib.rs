//! HIDE model orchestration (bible ch.06).
//!
//! The orchestrator is the shell-side fleet router over one or more
//! `hawking-serve` instances. It is HTTP/interface only — no GPU code here.
//!
//! The pipeline is: a request is **routed** ([`router`]) to a role; the role's
//! endpoint is resolved and called by the [`executor`]; a confidence-gated
//! [`escalation`] cascade retries up a stronger role when the cheap one is
//! uncertain; [`confidence`] supplies the self-consistency signal; [`grammar`]
//! enforces output envelopes via validate-and-retry; [`scheduler`] gates roles
//! against the machine's energy/thermal/RAM budget; and [`adapters`] selects
//! LoRA deltas per role/task. Live model calls cross the [`inference`] trait,
//! implemented over HTTP by [`http_client`].

pub mod adapters;
pub mod confidence;
pub mod difficulty;
pub mod escalation;
pub mod executor;
pub mod grammar;
pub mod http_client;
pub mod inference;
pub mod registry;
pub mod router;
pub mod sampler;
pub mod scheduler;
pub mod supervisor;
pub mod tool_spec_decode;

pub use adapters::{AdapterRegistry, AdapterSelection};
pub use confidence::{self_consistency_vote, AnswerNormalizer, VoteResult};
pub use escalation::{EscalationBudget, EscalationCascade, EscalationOutcome, SelfConsistencyProbe};
pub use executor::{Executor, HttpClientFactory};
pub use grammar::{GrammarMatcher, GrammarSpec, GrammarValidation, ShellGrammarCompiler};
pub use http_client::{GenerateRoute, HawkingHttpClient};
pub use inference::{InferenceClient, StubInferenceClient};
pub use registry::{default_hawking_local_roles, RoleRegistry};
pub use router::{RouteDecision, Router, SimpleRouter};
pub use scheduler::{Admission, ResourceSnapshot, Scheduler};
