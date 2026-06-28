//! Security infrastructure for HIDE (bible chapter 10).
//!
//! Backend-only pieces from ch.10: tamper-evident **blake3** hash-chain audit
//! with genesis salt + signed anchors (`audit`), secret **redaction** before
//! durability with a regex + entropy detector suite (`redaction`),
//! **encryption-at-rest** via AES-256-GCM AEAD with an OS-keychain-wrapped key
//! plus fail-closed layout validation (`storage`), and macOS **Seatbelt**
//! profile rendering + `sandbox-exec` spawning (`sandbox`).
//!
//! OS enforcement that needs a live host (the egress proxy, the microVM tier,
//! the Endpoint-Security reference monitor) remains a documented seam — see the
//! per-module docs. The pure security logic (chain math, detectors, AEAD,
//! profile rendering) is real and tested here.

pub mod audit;
pub mod redaction;
pub mod sandbox;
pub mod storage;

pub use audit::{
    chain_hash, compute_event_chain, compute_event_chain_salted, integrity_alarm_event,
    verify_event_chain, verify_event_chain_salted, verify_with_anchors, AnchorSigner,
    AnchoredVerification, ChainAnchor, ChainAuditReport, EventChainAuditor, IntegrityAlarmKind,
    CHAIN_HASH_LEN,
};
pub use redaction::{
    shannon_entropy, JsonRedactionReport, PatternDetector, Redaction, RedactionReport, Redactor,
};
pub use sandbox::{
    build_sandbox_exec_command, default_workspace_profile, emit_grant_profile,
    render_macos_seatbelt, render_macos_seatbelt_with, sandbox_exec_available, spawn_under_sandbox,
    RenderedSandboxProfile, SandboxRenderOptions, SandboxedCommand,
};
pub use storage::{
    ensure_and_validate_layout, validate_layout, AtRestCipher, AtRestPolicy, EncryptedSegment,
    FileWrapKeyStore, LayoutValidation, WrapKeyStore, WrappedWdk, NONCE_LEN, WDK_LEN,
};

#[cfg(feature = "os-keychain")]
pub use storage::KeychainWrapKeyStore;
