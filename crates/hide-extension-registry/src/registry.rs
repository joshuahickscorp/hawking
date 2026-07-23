//! The unified capability registry.
//!
//! One [`Registry`] holds every kind of capability behind one ABI. It enforces
//! the invariants the rest of HIDE relies on:
//! - a capability may not register an effect it does not declare (undeclared
//!   effects that its policies or scopes imply are a hard registration error),
//! - a capability that executes or spawns a process must declare sandbox
//!   isolation,
//! - ids are unique,
//! - a pinned id must match its pinned version and provenance,
//! - the full schema is disclosed only on request,
//! - revoked capabilities disappear from resolution and cannot be loaded.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{EffectList, RegistryError, Result};
use crate::index::CompactEntry;
use crate::manifest::{CapabilityKind, CapabilityManifest, Effect, FullSchema, Scope, SchemaRef};

/// A pin binds an id to an expected version and/or provenance. Any field left
/// `None` is unconstrained. Registration that names a pinned id must satisfy
/// every constrained field.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinSpec {
    pub version: Option<String>,
    pub source: Option<String>,
    pub commit: Option<String>,
}

impl PinSpec {
    /// Explain the first mismatch between this pin and a manifest, or `None` if
    /// the manifest satisfies the pin.
    fn mismatch(&self, m: &CapabilityManifest) -> Option<String> {
        if let Some(v) = &self.version {
            if &m.version != v {
                return Some(format!("version {:?} != pinned {:?}", m.version, v));
            }
        }
        if let Some(s) = &self.source {
            if &m.provenance.source != s {
                return Some(format!(
                    "provenance source {:?} != pinned {:?}",
                    m.provenance.source, s
                ));
            }
        }
        if let Some(c) = &self.commit {
            match &m.provenance.commit {
                Some(mc) if mc == c => {}
                Some(mc) => return Some(format!("provenance commit {mc:?} != pinned {c:?}")),
                None => return Some(format!("provenance commit missing, pinned {c:?}")),
            }
        }
        None
    }
}

/// A resolution request. Every field is optional or empty by default; an
/// all-default query matches every active capability.
#[derive(Debug, Clone, Default)]
pub struct ResolveQuery {
    /// Free text; whitespace-split keywords are matched against id and
    /// description to rank relevance. Does not filter, only ranks.
    pub task: Option<String>,
    /// Restrict to one kind.
    pub kind: Option<CapabilityKind>,
    /// Restrict to capabilities offered to this role.
    pub role: Option<String>,
    /// Every listed scope must be covered by the capability's declared scopes.
    pub required_scopes: Vec<Scope>,
}

impl ResolveQuery {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn task(mut self, t: impl Into<String>) -> Self {
        self.task = Some(t.into());
        self
    }
    pub fn kind(mut self, k: CapabilityKind) -> Self {
        self.kind = Some(k);
        self
    }
    pub fn role(mut self, r: impl Into<String>) -> Self {
        self.role = Some(r.into());
        self
    }
    pub fn require_scope(mut self, s: Scope) -> Self {
        self.required_scopes.push(s);
        self
    }
}

/// A candidate returned from [`Registry::resolve_for`], carrying its compact
/// entry plus the ranking factors that placed it. The list is returned already
/// sorted best-first, so callers can just take the head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedCandidate {
    pub entry: CompactEntry,
    /// Count of task keywords found in id or description.
    pub task_matches: usize,
    /// Whether the capability names the queried role explicitly.
    pub role_match: bool,
    /// Count of elevated (non-read) effects; fewer ranks higher.
    pub elevated_effects: usize,
    /// Schema load cost in tokens; cheaper ranks higher.
    pub schema_tokens: u32,
}

struct Entry {
    manifest: CapabilityManifest,
    revoked: bool,
}

/// The registry. Not `Clone` (it owns a monotonic schema-load counter used to
/// prove that resolution never eagerly loads schemas).
#[derive(Default)]
pub struct Registry {
    entries: BTreeMap<String, Entry>,
    pins: BTreeMap<String, PinSpec>,
    schema_loads: AtomicU64,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a capability. Fails on a duplicate id, on any effect the
    /// manifest's policies or scopes imply but the effects list omits, on empty
    /// required identity fields, and on a pin violation.
    pub fn register(&mut self, manifest: CapabilityManifest) -> Result<()> {
        if manifest.id.trim().is_empty() {
            return Err(RegistryError::InvalidManifest {
                id: manifest.id.clone(),
                detail: "id is empty".to_string(),
            });
        }
        if manifest.version.trim().is_empty() {
            return Err(RegistryError::InvalidManifest {
                id: manifest.id.clone(),
                detail: "version is empty".to_string(),
            });
        }
        if self.entries.contains_key(&manifest.id) {
            return Err(RegistryError::DuplicateId(manifest.id.clone()));
        }
        let missing = manifest.undeclared_effects();
        if !missing.is_empty() {
            return Err(RegistryError::UndeclaredEffects {
                id: manifest.id.clone(),
                missing: EffectList(missing),
            });
        }
        // Sandbox honesty. Registration is the one gate EVERY capability
        // crosses, including any manifest minted from an on-disk or otherwise
        // foreign declaration, so a manifest that claims it executes or spawns
        // a process while requiring no isolation is refused here rather than
        // trusted until call time. The builtin bridge already sets Subprocess
        // for exactly these effects; this makes that a checked invariant.
        if manifest
            .effects
            .iter()
            .any(|e| matches!(e, Effect::Execute | Effect::Process))
            && !manifest.sandbox.requires_isolation()
        {
            return Err(RegistryError::InvalidManifest {
                id: manifest.id.clone(),
                detail: "declares Execute or Process but requires no sandbox isolation".to_string(),
            });
        }
        if let Some(pin) = self.pins.get(&manifest.id) {
            if let Some(detail) = pin.mismatch(&manifest) {
                return Err(RegistryError::PinViolation {
                    id: manifest.id.clone(),
                    detail,
                });
            }
        }
        self.entries.insert(
            manifest.id.clone(),
            Entry {
                manifest,
                revoked: false,
            },
        );
        Ok(())
    }

    /// Pin an id to an expected version and/or provenance. If the id is already
    /// registered, the current manifest is validated against the pin
    /// immediately (a retroactive pin that the live manifest violates is
    /// rejected). Future registrations of this id must also satisfy the pin.
    pub fn pin(&mut self, id: impl Into<String>, pin: PinSpec) -> Result<()> {
        let id = id.into();
        if let Some(e) = self.entries.get(&id) {
            if let Some(detail) = pin.mismatch(&e.manifest) {
                return Err(RegistryError::PinViolation { id, detail });
            }
        }
        self.pins.insert(id, pin);
        Ok(())
    }

    /// Revoke a capability. It stays in the map (so its id cannot be silently
    /// reused) but is excluded from the index, from resolution, and from schema
    /// loads. Returns an error if the id was never registered.
    pub fn revoke(&mut self, id: &str) -> Result<()> {
        match self.entries.get_mut(id) {
            Some(e) => {
                e.revoked = true;
                Ok(())
            }
            None => Err(RegistryError::NotFound(id.to_string())),
        }
    }

    /// Whether an id is present (registered, possibly revoked).
    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Whether an id is present and revoked.
    pub fn is_revoked(&self, id: &str) -> bool {
        self.entries.get(id).map(|e| e.revoked).unwrap_or(false)
    }

    /// Number of active (non-revoked) capabilities.
    pub fn active_len(&self) -> usize {
        self.entries.values().filter(|e| !e.revoked).count()
    }

    /// The compact metadata index over active capabilities, ordered by id. This
    /// discloses only id, kind, description, and scopes; it never materializes a
    /// schema.
    pub fn index(&self) -> Vec<CompactEntry> {
        self.entries
            .values()
            .filter(|e| !e.revoked)
            .map(|e| CompactEntry::from_manifest(&e.manifest))
            .collect()
    }

    /// Resolve ranked candidates for a task, role, and scope requirement.
    ///
    /// Hard filters: kind (if set), role offering (if set), and full coverage of
    /// every required scope. Revoked capabilities never appear. Ranking, applied
    /// after filtering, is fully deterministic:
    /// 1. more matched task keywords first,
    /// 2. explicit role match first,
    /// 3. fewer elevated effects first (least privilege),
    /// 4. lower schema-token cost first,
    /// 5. id ascending as the final tie-break.
    pub fn resolve_for(&self, q: &ResolveQuery) -> Vec<RankedCandidate> {
        let keywords = split_keywords(q.task.as_deref());
        let mut out: Vec<RankedCandidate> = Vec::new();

        for e in self.entries.values() {
            if e.revoked {
                continue;
            }
            let m = &e.manifest;
            if let Some(k) = q.kind {
                if m.kind != k {
                    continue;
                }
            }
            if let Some(role) = &q.role {
                if !m.offered_to_role(role) {
                    continue;
                }
            }
            if !q.required_scopes.iter().all(|s| m.scope_allows(s)) {
                continue;
            }

            let haystack = format!("{} {}", m.id, m.description).to_lowercase();
            let task_matches = keywords.iter().filter(|k| haystack.contains(*k)).count();
            let role_match = q.role.as_deref().map(|r| m.declares_role(r)).unwrap_or(false);

            out.push(RankedCandidate {
                entry: CompactEntry::from_manifest(m),
                task_matches,
                role_match,
                elevated_effects: m.elevated_effect_count(),
                schema_tokens: m.context_cost.schema_tokens,
            });
        }

        out.sort_by(|a, b| {
            b.task_matches
                .cmp(&a.task_matches)
                .then(b.role_match.cmp(&a.role_match))
                .then(a.elevated_effects.cmp(&b.elevated_effects))
                .then(a.schema_tokens.cmp(&b.schema_tokens))
                .then(a.entry.id.cmp(&b.entry.id))
        });
        out
    }

    /// Load and parse the full input/output schema for a capability. This is the
    /// only path that touches the heavy schema text, and it bumps the schema
    /// load counter so callers can prove resolution stayed lazy. Fails if the id
    /// is unknown, if it is revoked, or if a raw schema fails to parse.
    pub fn load_full_schema(&self, id: &str) -> Result<FullSchema> {
        let e = self
            .entries
            .get(id)
            .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
        if e.revoked {
            return Err(RegistryError::Revoked(id.to_string()));
        }
        let m = &e.manifest;
        let input = parse_schema(&m.input_schema_ref, id, "input")?;
        let output = parse_schema(&m.output_schema_ref, id, "output")?;
        self.schema_loads.fetch_add(1, Ordering::Relaxed);
        Ok(FullSchema {
            input_uri: m.input_schema_ref.uri.clone(),
            output_uri: m.output_schema_ref.uri.clone(),
            input,
            output,
        })
    }

    /// How many full-schema loads have happened. Zero after any number of
    /// registrations, index builds, and resolutions: those never load a schema.
    pub fn schema_load_count(&self) -> u64 {
        self.schema_loads.load(Ordering::Relaxed)
    }

    // --- enforcement helpers ------------------------------------------------

    /// The declared effects of a capability.
    pub fn declared_effects(&self, id: &str) -> Result<Vec<Effect>> {
        Ok(self.active(id)?.effects.clone())
    }

    /// Whether a capability requires sandbox isolation.
    pub fn requires_sandbox(&self, id: &str) -> Result<bool> {
        Ok(self.active(id)?.sandbox.requires_isolation())
    }

    /// Whether a capability's declared scopes cover a requested scope.
    pub fn scope_allows(&self, id: &str, scope: &Scope) -> Result<bool> {
        Ok(self.active(id)?.scope_allows(scope))
    }

    /// The pinned identity a capability was registered under (version and
    /// provenance), for audit and pinning.
    pub fn version(&self, id: &str) -> Result<String> {
        Ok(self.active(id)?.version.clone())
    }

    /// The provenance of a capability.
    pub fn provenance(&self, id: &str) -> Result<crate::manifest::Provenance> {
        Ok(self.active(id)?.provenance.clone())
    }

    /// The kind of a capability.
    pub fn kind(&self, id: &str) -> Result<CapabilityKind> {
        Ok(self.active(id)?.kind)
    }

    /// The context cost (schema token estimate) of a capability.
    pub fn context_cost(&self, id: &str) -> Result<u32> {
        Ok(self.active(id)?.context_cost.schema_tokens)
    }

    /// Borrow the active (non-revoked) manifest for `id`.
    fn active(&self, id: &str) -> Result<&CapabilityManifest> {
        let e = self
            .entries
            .get(id)
            .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
        if e.revoked {
            return Err(RegistryError::Revoked(id.to_string()));
        }
        Ok(&e.manifest)
    }
}

fn split_keywords(task: Option<&str>) -> Vec<String> {
    match task {
        None => Vec::new(),
        Some(t) => t
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| !w.is_empty())
            .collect(),
    }
}

fn parse_schema(
    r: &SchemaRef,
    id: &str,
    which: &'static str,
) -> Result<Option<serde_json::Value>> {
    match &r.raw {
        None => Ok(None),
        Some(text) => {
            let v = serde_json::from_str(text).map_err(|source| RegistryError::Schema {
                id: id.to_string(),
                which,
                source,
            })?;
            Ok(Some(v))
        }
    }
}
