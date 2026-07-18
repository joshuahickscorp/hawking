//! Hawking Seed — the smallest stable object from which Gravity, Forge, Doctor, runtime, and
//! evidence are summoned. Seed owns authority + contracts; packs own optional implementation mass.
//!
//! One `Record` envelope, one transition engine, Gravity/Forge/Doctor as policies over the lifecycle,
//! a tiny runtime contract delegating model math to the default runtime pack (hawking-core).

pub mod doctor;
pub mod evidence;
pub mod forge;
pub mod gravity;
pub mod pack;
pub mod record;
pub mod runtime;
pub mod state;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("seal: {0}")]
    Seal(String),
    #[error("transition: {0}")]
    Transition(String),
    #[error("gravity: {0}")]
    Gravity(String),
    #[error("pack: {0}")]
    Pack(String),
    #[error("runtime: {0}")]
    Runtime(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
