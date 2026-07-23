//! Capability manifest types.
//!
//! A [`CapabilityManifest`] is the single declaration a provider files for any
//! extension that HIDE can call: a tool, a skill, a plugin, a hook, an MCP or
//! ACP endpoint, a subagent, a rule, a command, an oracle, a browser action, or
//! an integration. The manifest carries lightweight metadata (indexed and
//! disclosed up front) and references to heavier input/output schemas that are
//! only materialized on demand. It also states, up front and checkably, what the
//! capability is allowed to do: its effects, scopes, sandbox requirement,
//! network policy, secret policy, and provenance.

use serde::{Deserialize, Serialize};

/// The category of an extension. One registry holds all of them so that
/// resolution, effect accounting, and progressive disclosure work the same way
/// regardless of how a capability is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityKind {
    Tool,
    Skill,
    Plugin,
    Hook,
    Mcp,
    Acp,
    Subagent,
    Rule,
    Command,
    Oracle,
    Browser,
    Integration,
}

impl CapabilityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CapabilityKind::Tool => "tool",
            CapabilityKind::Skill => "skill",
            CapabilityKind::Plugin => "plugin",
            CapabilityKind::Hook => "hook",
            CapabilityKind::Mcp => "mcp",
            CapabilityKind::Acp => "acp",
            CapabilityKind::Subagent => "subagent",
            CapabilityKind::Rule => "rule",
            CapabilityKind::Command => "command",
            CapabilityKind::Oracle => "oracle",
            CapabilityKind::Browser => "browser",
            CapabilityKind::Integration => "integration",
        }
    }
}

/// A single side effect a capability may produce. Effects are declared, not
/// inferred at call time, so a planner can reason about a capability before
/// invoking it. Everything except [`Effect::Read`] counts as elevated for
/// least-privilege ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Effect {
    Read,
    Write,
    /// Mutates version-control state (commits, refs, worktrees). A specialized
    /// write whose blast radius is the repository's git history and working
    /// tree layout, not just a single file's bytes.
    GitMutation,
    Execute,
    /// Spawns or controls an operating-system process. Distinct from
    /// [`Effect::Execute`]: a tool can execute one fixed, bounded command
    /// without the open-world reach to spawn and manage arbitrary processes.
    Process,
    Network,
    SecretAccess,
    ExternalMutation,
    Irreversible,
    Privileged,
}

impl Effect {
    pub fn as_str(&self) -> &'static str {
        match self {
            Effect::Read => "Read",
            Effect::Write => "Write",
            Effect::GitMutation => "GitMutation",
            Effect::Execute => "Execute",
            Effect::Process => "Process",
            Effect::Network => "Network",
            Effect::SecretAccess => "SecretAccess",
            Effect::ExternalMutation => "ExternalMutation",
            Effect::Irreversible => "Irreversible",
            Effect::Privileged => "Privileged",
        }
    }

    /// Everything above a plain read is elevated. Resolution ranks lower-effect
    /// capabilities first so a task is served by the least-privilege option.
    pub fn is_elevated(&self) -> bool {
        !matches!(self, Effect::Read)
    }
}

/// A resource or role scope a capability is bound to. Scopes are matched with
/// coverage rules (see [`Scope::covers`]) so a broad grant satisfies a narrow
/// request without listing every leaf.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// The whole working repository.
    Repo,
    /// A filesystem path prefix (a parent path covers its descendants).
    Filesystem(String),
    /// A network host (`"*"` covers any host).
    Network(String),
    /// A named environment variable (`"*"` covers any).
    Env(String),
    /// A named secret (`"*"` covers any).
    Secret(String),
    /// An operator role this capability is offered to.
    Role(String),
    /// An opaque capability-defined scope, matched by equality.
    Custom(String),
}

impl Scope {
    /// Whether a held scope (`self`) covers a requested scope (`want`).
    pub fn covers(&self, want: &Scope) -> bool {
        use Scope::*;
        match (self, want) {
            (Repo, Repo) => true,
            (Filesystem(a), Filesystem(b)) => path_prefix_covers(a, b),
            (Network(a), Network(b)) => a == "*" || a == b,
            (Env(a), Env(b)) => a == "*" || a == b,
            (Secret(a), Secret(b)) => a == "*" || a == b,
            (Role(a), Role(b)) => a == b,
            (Custom(a), Custom(b)) => a == b,
            _ => false,
        }
    }
}

/// True when path `a` is a prefix directory of path `b` (component aware).
fn path_prefix_covers(a: &str, b: &str) -> bool {
    let a = a.trim_end_matches('/');
    let b = b.trim_end_matches('/');
    if a.is_empty() {
        // An empty (root) grant covers everything.
        return true;
    }
    b == a || b.starts_with(&format!("{a}/"))
}

/// How isolated a capability must run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SandboxReq {
    /// Runs in-process with no added isolation.
    #[default]
    None,
    /// Must run in an isolated subprocess.
    Subprocess,
    /// Must run in a container.
    Container,
    /// Must run in a virtual machine.
    Vm,
}

impl SandboxReq {
    pub fn requires_isolation(&self) -> bool {
        !matches!(self, SandboxReq::None)
    }
}

/// What network access a capability is permitted. Denied by default: a
/// capability that wants the network has to say so, and saying so implies the
/// [`Effect::Network`] declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkPolicy {
    /// No network access.
    Deny,
    /// Access limited to the listed hosts.
    AllowList(Vec<String>),
    /// Unrestricted network access.
    Any,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        NetworkPolicy::Deny
    }
}

impl NetworkPolicy {
    pub fn grants_network(&self) -> bool {
        !matches!(self, NetworkPolicy::Deny)
    }
}

/// What secret material a capability may read. None by default; anything else
/// implies the [`Effect::SecretAccess`] declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretPolicy {
    /// No secret access.
    None,
    /// Access limited to the listed named secrets.
    Named(Vec<String>),
    /// Access to any secret.
    Any,
}

impl Default for SecretPolicy {
    fn default() -> Self {
        SecretPolicy::None
    }
}

impl SecretPolicy {
    pub fn grants_secret(&self) -> bool {
        !matches!(self, SecretPolicy::None)
    }
}

/// The cost of pulling a capability's full schema into context. Progressive
/// disclosure trades this off: the compact index is cheap, and the caller pays
/// `schema_tokens` only when it actually loads the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ContextCost {
    pub schema_tokens: u32,
}

/// Where a capability came from, for pinning and audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub license: String,
}

/// A reference to a schema document. The `uri` is always present and cheap; the
/// `raw` payload is the heavy schema text that the registry only parses when
/// [`crate::Registry::load_full_schema`] is called. Keeping the raw text out of
/// the compact index and out of every accessor except the explicit load is what
/// makes disclosure progressive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaRef {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl SchemaRef {
    pub fn new(uri: impl Into<String>) -> Self {
        SchemaRef {
            uri: uri.into(),
            raw: None,
        }
    }

    pub fn with_raw(uri: impl Into<String>, raw: impl Into<String>) -> Self {
        SchemaRef {
            uri: uri.into(),
            raw: Some(raw.into()),
        }
    }
}

/// The full, on-demand schema view. Produced only by an explicit load so the
/// heavy JSON never rides along with metadata resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullSchema {
    pub input_uri: String,
    pub output_uri: String,
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
}

/// The single unified declaration for any capability the registry can serve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub id: String,
    pub version: String,
    pub kind: CapabilityKind,
    pub provider: String,
    pub license: String,
    /// Short human description, surfaced in the compact index for disclosure and
    /// used as a haystack when ranking against a task query.
    #[serde(default)]
    pub description: String,
    pub effects: Vec<Effect>,
    pub input_schema_ref: SchemaRef,
    pub output_schema_ref: SchemaRef,
    pub scopes: Vec<Scope>,
    #[serde(default)]
    pub sandbox: SandboxReq,
    #[serde(default)]
    pub network: NetworkPolicy,
    #[serde(default)]
    pub secrets: SecretPolicy,
    #[serde(default)]
    pub context_cost: ContextCost,
    pub provenance: Provenance,
}

impl CapabilityManifest {
    /// A minimal manifest with denied network, no secrets, no sandbox, and a
    /// single read effect. Tests and callers override fields from here.
    pub fn new(
        id: impl Into<String>,
        version: impl Into<String>,
        kind: CapabilityKind,
        provider: impl Into<String>,
    ) -> Self {
        let id = id.into();
        CapabilityManifest {
            input_schema_ref: SchemaRef::new(format!("schema://{id}/input")),
            output_schema_ref: SchemaRef::new(format!("schema://{id}/output")),
            id,
            version: version.into(),
            kind,
            provider: provider.into(),
            license: "MIT".to_string(),
            description: String::new(),
            effects: vec![Effect::Read],
            scopes: Vec::new(),
            sandbox: SandboxReq::None,
            network: NetworkPolicy::Deny,
            secrets: SecretPolicy::None,
            context_cost: ContextCost::default(),
            provenance: Provenance {
                source: "builtin".to_string(),
                commit: None,
                license: "MIT".to_string(),
            },
        }
    }

    /// The effects that this manifest's policies and scopes require it to hold.
    /// A network policy that grants access requires [`Effect::Network`]; a secret
    /// policy that grants access requires [`Effect::SecretAccess`]; a network or
    /// secret scope requires the matching effect. Order is stable and de-duped.
    pub fn implied_effects(&self) -> Vec<Effect> {
        let mut out: Vec<Effect> = Vec::new();
        let push = |e: Effect, out: &mut Vec<Effect>| {
            if !out.contains(&e) {
                out.push(e);
            }
        };
        if self.network.grants_network() {
            push(Effect::Network, &mut out);
        }
        if self.secrets.grants_secret() {
            push(Effect::SecretAccess, &mut out);
        }
        for s in &self.scopes {
            match s {
                Scope::Network(_) => push(Effect::Network, &mut out),
                Scope::Secret(_) => push(Effect::SecretAccess, &mut out),
                _ => {}
            }
        }
        out
    }

    /// Implied effects that the manifest fails to declare. Empty means the
    /// declaration is complete; a non-empty result is a registration blocker.
    pub fn undeclared_effects(&self) -> Vec<Effect> {
        self.implied_effects()
            .into_iter()
            .filter(|e| !self.effects.contains(e))
            .collect()
    }

    /// Whether this manifest's scopes cover a requested scope.
    pub fn scope_allows(&self, want: &Scope) -> bool {
        self.scopes.iter().any(|held| held.covers(want))
    }

    /// Whether this manifest is offered to a given role. A manifest with no role
    /// scope is offered to every role; one with role scopes is offered only to
    /// the roles it names.
    pub fn offered_to_role(&self, role: &str) -> bool {
        let mut has_role_scope = false;
        for s in &self.scopes {
            if let Scope::Role(r) = s {
                has_role_scope = true;
                if r == role {
                    return true;
                }
            }
        }
        !has_role_scope
    }

    /// Whether this manifest explicitly names a role scope for `role`.
    pub fn declares_role(&self, role: &str) -> bool {
        self.scopes
            .iter()
            .any(|s| matches!(s, Scope::Role(r) if r == role))
    }

    /// Count of elevated (non-read) effects, used for least-privilege ranking.
    pub fn elevated_effect_count(&self) -> usize {
        self.effects.iter().filter(|e| e.is_elevated()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filesystem_prefix_coverage() {
        let held = Scope::Filesystem("src".to_string());
        assert!(held.covers(&Scope::Filesystem("src/lib.rs".to_string())));
        assert!(held.covers(&Scope::Filesystem("src".to_string())));
        assert!(!held.covers(&Scope::Filesystem("tests".to_string())));
        // A sibling that merely shares a prefix string is not covered.
        assert!(!held.covers(&Scope::Filesystem("srcextra".to_string())));
    }

    #[test]
    fn network_wildcard_coverage() {
        let any = Scope::Network("*".to_string());
        assert!(any.covers(&Scope::Network("example.com".to_string())));
        let one = Scope::Network("example.com".to_string());
        assert!(one.covers(&Scope::Network("example.com".to_string())));
        assert!(!one.covers(&Scope::Network("evil.com".to_string())));
    }

    #[test]
    fn implied_effects_from_policies_and_scopes() {
        let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
        m.network = NetworkPolicy::Any;
        m.secrets = SecretPolicy::Named(vec!["token".to_string()]);
        m.scopes = vec![Scope::Network("api.example.com".to_string())];
        let implied = m.implied_effects();
        assert!(implied.contains(&Effect::Network));
        assert!(implied.contains(&Effect::SecretAccess));
        // De-dup: Network is implied by both policy and scope but appears once.
        assert_eq!(
            implied.iter().filter(|e| **e == Effect::Network).count(),
            1
        );
    }

    #[test]
    fn undeclared_effects_detected() {
        let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
        m.network = NetworkPolicy::Any;
        // effects still only Read, so Network is undeclared.
        assert_eq!(m.undeclared_effects(), vec![Effect::Network]);
        m.effects.push(Effect::Network);
        assert!(m.undeclared_effects().is_empty());
    }

    #[test]
    fn role_offering() {
        let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
        // No role scope: offered to any role.
        assert!(m.offered_to_role("reviewer"));
        m.scopes = vec![Scope::Role("admin".to_string())];
        assert!(m.offered_to_role("admin"));
        assert!(!m.offered_to_role("reviewer"));
        assert!(m.declares_role("admin"));
    }
}
