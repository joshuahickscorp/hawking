//! Errors for the verification plane. Only the filesystem-facing paths (a walked
//! directory scan) can fail; the in-memory analysis over source strings is
//! infallible and returns findings directly.

use thiserror::Error;

/// A failure while walking or reading a directory of Rust source.
#[derive(Debug, Error)]
pub enum VerifyError {
    /// The directory walker itself failed (permissions, a vanished entry, a
    /// symlink loop). Carries a rendered message so the public API never leaks
    /// the underlying `walkdir` error type.
    #[error("directory walk failed under {root}: {message}")]
    Walk { root: String, message: String },

    /// A source file could not be read as UTF-8 text.
    #[error("failed to read {path}: {message}")]
    Read { path: String, message: String },
}

/// Result alias for the fallible directory-facing surface of this crate.
pub type Result<T> = std::result::Result<T, VerifyError>;
