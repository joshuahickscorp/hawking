//! Hawking Seed — Candidate C: the Event Horizon engine.
//!
//! Candidate C's thesis: a tiny Hawking runtime can execute weights in their COMPRESSED representation
//! directly — memory-mapped, dequantized only at bounded tile scale inside the operators — using the
//! M3 Ultra where it helps, with NO full-model f32 expansion and NO dense shadow. It shares Candidate
//! B's proven numerics and authority but is a materially different runtime: direct-quant operators, a
//! Metal backend, a sub-bit compact operator, and an MoE-ready IR with a bounded 120B F2 bridge.
//!
//! No hawking-core, no predecessor engine, no subprocess.

// shared authority (reused from Candidate B: the seal/identity/control/evidence core)
pub mod evidence;
pub mod gravity;
pub mod pack;
pub mod record;
pub mod state;

// direct-compact runtime
pub mod adapter;
pub mod cpu;
pub mod gguf;
pub mod ir;
pub mod metal;
pub mod model;
pub mod gptoss;
pub mod gravity_run;
pub mod mxfp4;
pub mod quant;
pub mod safetensors;
pub mod tokenizer;

// sub-bit direct execution + Doctor rescue + the bounded MoE / 120B F2 bridge.
// (Candidate C's Doctor is subbit::doctor_rescue, executed through the compact operator — it does NOT
// reuse Candidate A/B's disconnected int8 forge/doctor fixture, per the directive.)
pub mod f2;
pub mod subbit;

// Absorbed hawking-packs nucleus: the ONE Pack ABI's capability providers (adapters, forge, doctor, metal,
// speculation, validation, experiment) plus the one registry/verifier/profiles, all orbiting this Seed's
// own authority (record/evidence/state/pack/gravity/ir/subbit). No external crate; internal modules only.
pub mod providers;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("seal: {0}")]
    Seal(String),
    #[error("transition: {0}")]
    Transition(String),
    #[error("pack: {0}")]
    Pack(String),
    #[error("pack {pack} TAMPERED: {path} sha {got} != declared {declared}")]
    Tamper { pack: String, path: String, got: String, declared: String },
    #[error("registry: {0}")]
    Registry(String),
    #[error("adapter: {0}")]
    Adapter(String),
    #[error("provider: {0}")]
    Provider(String),
    #[error("gravity: {0}")]
    Gravity(String),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("model: {0}")]
    Model(String),
    #[error("runtime: {0}")]
    Runtime(String),
    #[error("metal: {0}")]
    Metal(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
