//! Connector ABI: the single declaration every YOU connector family files.
//!
//! Every family (implemented or merely declared) fills the same fields: read/
//! write capabilities, auth method, scopes, object types, sync mode, offline
//! cache, rate limits, effect classes, revocation, and audit receipts. A filled
//! ABI is not an implementation — only [`ImplementationStatus::Implemented`]
//! families are constructible.

use serde::{Deserialize, Serialize};

/// Stable family id string (e.g. `local_folder`, `rss`, `github`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FamilyId(pub String);

impl FamilyId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FamilyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for FamilyId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for FamilyId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Whether this family has a real constructor or is ABI-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationStatus {
    /// End-to-end implementation exists (fixture-backed is fine; network is not required).
    Implemented,
    /// ABI filled; construction is refused.
    Declared,
}

impl ImplementationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Declared => "declared",
        }
    }
}

/// Read capability declaration. Default for every connector is at least list/
/// fetch of its object types; a family with `reads: false` is inert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadCapability {
    /// May list objects in scope.
    pub list: bool,
    /// May fetch object content/metadata.
    pub fetch: bool,
    /// May search within the connector's object space.
    pub search: bool,
}

impl ReadCapability {
    pub fn list_and_fetch() -> Self {
        Self {
            list: true,
            fetch: true,
            search: false,
        }
    }
    pub fn full() -> Self {
        Self {
            list: true,
            fetch: true,
            search: true,
        }
    }
    pub fn none() -> Self {
        Self {
            list: false,
            fetch: false,
            search: false,
        }
    }
}

/// Write capability declaration. Absence (or all-false) means the connector is
/// read-only at the type boundary — there is no `ConnectorWrite` impl to call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteCapability {
    pub create: bool,
    pub update: bool,
    pub delete: bool,
}

impl WriteCapability {
    pub fn none() -> Self {
        Self {
            create: false,
            update: false,
            delete: false,
        }
    }
    pub fn full() -> Self {
        Self {
            create: true,
            update: true,
            delete: true,
        }
    }
    pub fn is_writable(&self) -> bool {
        self.create || self.update || self.delete
    }
}

/// How the connector authenticates. No ambient credentials: auth is always
/// bound to an explicit account handle minted by the account store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// No credentials; path/scope is the account (local folders, fixture RSS).
    None,
    /// OAuth2 (authorization code / device / etc.). Not implemented here.
    OAuth2 {
        authorization_url: String,
        token_url: String,
    },
    /// Personal access token or API key held by the account store.
    ApiToken,
    /// OS keychain / platform secret for local services.
    LocalSecret,
    /// MCP session negotiated out of band.
    McpSession,
    /// Generic bearer / custom scheme declared by the family.
    Custom(String),
}

/// A permission scope the connector may request or hold.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectorScope {
    pub name: String,
    pub description: String,
    /// Whether this scope implies write (elevated).
    pub write: bool,
}

impl ConnectorScope {
    pub fn read(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            write: false,
        }
    }
    pub fn write(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            write: true,
        }
    }
}

/// Object types the connector can surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectType {
    pub name: String,
    pub description: String,
}

impl ObjectType {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Incremental sync support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    /// Full re-list only.
    FullOnly,
    /// Cursor / page token incremental.
    Cursor,
    /// Timestamp / etag based.
    Timestamp,
    /// Provider-specific delta tokens.
    DeltaToken,
}

/// How the connector learns about remote changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeTransport {
    /// No push; caller polls.
    Polling { min_interval_secs: u64 },
    /// Webhook / push notifications (declared only; not implemented here).
    Webhook { verification: String },
    /// Both polling and webhooks supported.
    PollingAndWebhook {
        min_interval_secs: u64,
        verification: String,
    },
    /// Local filesystem watch (inotify / FSEvents style).
    LocalWatch,
    /// Not applicable (static fixture, one-shot read).
    None,
}

/// Offline cache policy declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineCache {
    pub supported: bool,
    /// Max bytes the connector may retain offline (0 = none).
    pub max_bytes: u64,
    /// Whether cache may hold content bodies or only metadata.
    pub bodies: bool,
}

impl OfflineCache {
    pub fn none() -> Self {
        Self {
            supported: false,
            max_bytes: 0,
            bodies: false,
        }
    }
    pub fn metadata(max_bytes: u64) -> Self {
        Self {
            supported: true,
            max_bytes,
            bodies: false,
        }
    }
    pub fn full(max_bytes: u64) -> Self {
        Self {
            supported: true,
            max_bytes,
            bodies: true,
        }
    }
}

/// Rate limit declaration (honest ceiling; not a live limiter).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst: u32,
    pub notes: String,
}

impl RateLimit {
    pub fn local() -> Self {
        Self {
            requests_per_minute: 600,
            burst: 60,
            notes: "local / fixture; no remote provider limit".into(),
        }
    }
    pub fn remote(rpm: u32, burst: u32, notes: impl Into<String>) -> Self {
        Self {
            requests_per_minute: rpm,
            burst,
            notes: notes.into(),
        }
    }
}

/// Effect classes a connector operation may produce. Writes are always an
/// elevated effect and must leave an audit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectClass {
    Read,
    Write,
    Delete,
    ExternalMutation,
    SecretAccess,
    Network,
}

impl EffectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Delete => "delete",
            Self::ExternalMutation => "external_mutation",
            Self::SecretAccess => "secret_access",
            Self::Network => "network",
        }
    }
    pub fn is_elevated(self) -> bool {
        !matches!(self, Self::Read)
    }
}

/// Revocation semantics. Real revocation invalidates handles and fails in-flight
/// work closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationPolicy {
    /// Handles are generation-checked; revoke bumps generation.
    pub invalidates_handles: bool,
    /// In-flight ops re-check handle before completing.
    pub fail_closed_in_flight: bool,
    /// Remote token revoke is attempted when the provider supports it.
    pub remote_revoke: bool,
    pub notes: String,
}

impl RevocationPolicy {
    pub fn real_local() -> Self {
        Self {
            invalidates_handles: true,
            fail_closed_in_flight: true,
            remote_revoke: false,
            notes: "local account store generation bump; fail closed".into(),
        }
    }
    pub fn real_with_remote() -> Self {
        Self {
            invalidates_handles: true,
            fail_closed_in_flight: true,
            remote_revoke: true,
            notes: "local generation bump plus remote token revoke when implemented".into(),
        }
    }
}

/// Audit receipt policy for connector operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditPolicy {
    /// Every write must leave a receipt.
    pub write_receipts_required: bool,
    /// Reads may optionally leave receipts (list/fetch).
    pub read_receipts: bool,
    pub notes: String,
}

impl AuditPolicy {
    pub fn writes_required() -> Self {
        Self {
            write_receipts_required: true,
            read_receipts: false,
            notes: "every write is an effect that leaves a receipt; no silent execution".into(),
        }
    }
}

/// The full connector ABI declaration for one family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorAbi {
    pub family_id: FamilyId,
    pub display_name: String,
    pub description: String,
    pub status: ImplementationStatus,
    pub read: ReadCapability,
    pub write: WriteCapability,
    pub auth: AuthMethod,
    pub scopes: Vec<ConnectorScope>,
    pub object_types: Vec<ObjectType>,
    pub sync: SyncMode,
    pub change_transport: ChangeTransport,
    pub offline_cache: OfflineCache,
    pub rate_limit: RateLimit,
    pub effect_classes: Vec<EffectClass>,
    pub revocation: RevocationPolicy,
    pub audit: AuditPolicy,
    /// Honest notes: what is real vs declared-only.
    pub honesty_notes: String,
}

impl ConnectorAbi {
    /// True when the ABI claims write capability.
    pub fn declares_write(&self) -> bool {
        self.write.is_writable()
    }

    /// True when construction is allowed.
    pub fn is_implemented(&self) -> bool {
        self.status == ImplementationStatus::Implemented
    }

    /// Validate internal consistency of the declaration.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        if self.family_id.0.is_empty() {
            errs.push("family_id empty".into());
        }
        if self.declares_write() && !self.effect_classes.contains(&EffectClass::Write) {
            errs.push(format!(
                "{}: write capability requires EffectClass::Write",
                self.family_id
            ));
        }
        if self.declares_write() && !self.audit.write_receipts_required {
            errs.push(format!(
                "{}: write capability requires audit.write_receipts_required",
                self.family_id
            ));
        }
        if !self.revocation.invalidates_handles {
            errs.push(format!(
                "{}: revocation must invalidate handles",
                self.family_id
            ));
        }
        if !self.revocation.fail_closed_in_flight {
            errs.push(format!(
                "{}: revocation must fail closed in-flight",
                self.family_id
            ));
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}
