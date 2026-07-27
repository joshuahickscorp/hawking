//! Errors surfaced by the browser evidence and verification layer.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum BrowserError {
    /// The replay ran off the end of the recorded session: a driver call was
    /// made with no further recorded step to satisfy it.
    #[error("replay exhausted: no recorded step remains for the requested {requested} call")]
    ReplayExhausted { requested: &'static str },

    /// A driver call did not match the next recorded step. The replay driver is
    /// a strict contract: the caller must drive the exact recorded sequence.
    #[error("replay mismatch at step {index}: expected {expected}, got a {requested} call ({detail})")]
    ReplayMismatch {
        index: usize,
        requested: &'static str,
        expected: String,
        detail: String,
    },

    /// An observer method was called before any step had been played.
    #[error("no current step: play a navigate/click/fill step before reading evidence")]
    NoCurrentStep,

    /// The current step did not capture the requested piece of evidence.
    #[error("the current step captured no {what}")]
    MissingEvidence { what: &'static str },

    /// A selection could not be resolved to a node in the given snapshot.
    #[error("could not resolve selection {selector:?} to a DOM node")]
    UnresolvedSelection { selector: String },

    #[error("serialization error: {0}")]
    Serde(String),
}

impl From<serde_json::Error> for BrowserError {
    fn from(e: serde_json::Error) -> Self {
        BrowserError::Serde(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, BrowserError>;
