//! Tool-call spec-decode: the "small spec-decode layer for tools" (see
//! `docs/plans/agentic_tool_system_2026_07_11.md`, Phases 2 and 3).
//!
//! A tool call is 60-80 percent deterministic given the schema, so most of its
//! tokens can be emitted with no forward pass at all. This module provides the
//! two training-free, lossless primitives that exploit that, at the string level
//! (the runtime maps them onto its per-token logit mask and its verifier):
//!
//! * [`ToolCallGrammar`] - jump-forward. It knows the canonical tool-call
//!   envelope `{"name": "<tool>", "arguments": {...}}` and the registered tool
//!   names, and reports the continuation that is the ONLY legal one from a given
//!   state: the opening scaffolding, the shared prefix of the still-consistent
//!   tool names, and the full skeleton once the tool is fixed. Emitting those is
//!   free and cannot change the sampled distribution (they were forced anyway).
//! * [`PromptLookup`] - draft argument values that the model is copying out of
//!   context (a path it just read, a symbol from the diff) by matching the tail
//!   of what it has generated against the context and proposing the continuation.
//!   The target still verifies each drafted token, so acceptance is lossless.
//!
//! Both are pure and fully unit-tested with no model. The runtime consumes them:
//! the grammar feeds `mask_logits` / a fast-forward emit, and the lookup feeds
//! the existing `speculate` verifier.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// schema-aware tool-call grammar (jump-forward)
// ---------------------------------------------------------------------------

/// The minimal schema shape the grammar needs: a tool name and its required
/// argument keys (in declared order). Derived from a `ToolSpec.input_schema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    #[serde(default)]
    pub required_keys: Vec<String>,
    /// Total number of DECLARED argument properties (the schema's `properties`
    /// map), or 0 when unknown. Needed to tell whether a single required key is
    /// the ONLY possible first key. Without it we cannot safely jump the key.
    #[serde(default)]
    pub declarable_keys: usize,
    /// Whether the schema forbids undeclared keys (`additionalProperties:false`).
    /// Only a closed schema can guarantee no other key appears first.
    #[serde(default)]
    pub closed: bool,
}

impl ToolSchema {
    /// Convenience constructor. `declarable_keys`/`closed` default to
    /// unknown/false, so key jump-forward stays OFF unless the shape is known via
    /// [`ToolSchema::from_input_schema`] or [`ToolSchema::with_shape`]. This keeps
    /// the losslessness invariant safe by default.
    pub fn new(name: impl Into<String>, required_keys: Vec<String>) -> Self {
        Self {
            name: name.into(),
            required_keys,
            declarable_keys: 0,
            closed: false,
        }
    }

    /// Declare the full argument shape (total declared property count + whether the
    /// schema is closed) so the grammar can decide when key jump-forward is safe.
    pub fn with_shape(mut self, declarable_keys: usize, closed: bool) -> Self {
        self.declarable_keys = declarable_keys;
        self.closed = closed;
        self
    }

    /// Build from a JSON-Schema-ish object, reading `required`, the full
    /// `properties` set, and `additionalProperties`, as carried on every
    /// `ToolSpec.input_schema`.
    pub fn from_input_schema(name: impl Into<String>, schema: &Value) -> Self {
        let required_keys = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let declarable_keys = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .map(|p| p.len())
            .unwrap_or(0);
        let closed = schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false);
        Self {
            name: name.into(),
            required_keys,
            declarable_keys,
            closed,
        }
    }
}

/// What the grammar reports about the tool-name field given the chars typed so
/// far (the characters after the opening quote of the name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameJump {
    /// The continuation that is the only legal one: the shared prefix of every
    /// registered name still consistent with `typed`, beyond what is typed.
    pub forced: String,
    /// `Some(name)` once `typed + forced` uniquely identifies a full tool name.
    pub resolved: Option<String>,
    /// True when `typed` is a prefix of no registered tool (a dead branch the
    /// constraint would have prevented; surfaced so the caller can reject).
    pub dead: bool,
}

/// The canonical tool-call grammar over a fixed set of registered tools.
#[derive(Debug, Clone)]
pub struct ToolCallGrammar {
    schemas: Vec<ToolSchema>,
}

impl ToolCallGrammar {
    /// Build from the registered tool schemas. Names are de-duplicated and sorted
    /// so common-prefix computation is stable.
    pub fn new(mut schemas: Vec<ToolSchema>) -> Self {
        schemas.sort_by(|a, b| a.name.cmp(&b.name));
        schemas.dedup_by(|a, b| a.name == b.name);
        Self { schemas }
    }

    /// The registered tool names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.schemas.iter().map(|s| s.name.as_str()).collect()
    }

    /// The forced opening scaffolding: from an empty output, this exact string is
    /// the only legal start of a tool call, so it can be emitted with no forward
    /// pass. (The runtime uses the whitespace-canonical form.)
    pub fn envelope_prefix(&self) -> &'static str {
        "{\"name\": \""
    }

    /// The closing brace of the outer envelope, forced once the arguments object
    /// is complete.
    pub fn envelope_suffix(&self) -> &'static str {
        "}"
    }

    /// Jump-forward within the tool-name field. Given the name characters typed so
    /// far, return the forced shared continuation and, once unambiguous, the
    /// resolved name.
    pub fn name_jump(&self, typed: &str) -> NameJump {
        let consistent: Vec<&str> = self
            .schemas
            .iter()
            .map(|s| s.name.as_str())
            .filter(|n| n.starts_with(typed))
            .collect();
        if consistent.is_empty() {
            return NameJump {
                forced: String::new(),
                resolved: None,
                dead: true,
            };
        }
        let lcp = longest_common_prefix(&consistent);
        let forced = lcp[typed.len().min(lcp.len())..].to_string();
        let resolved = if consistent.len() == 1 {
            Some(consistent[0].to_string())
        } else {
            None
        };
        NameJump {
            forced,
            resolved,
            dead: false,
        }
    }

    /// The maximal deterministic skeleton once the tool is known: the whole
    /// envelope up to the point where free content (the first argument value)
    /// begins. This is the single biggest jump-forward win: given the resolved
    /// tool, the runtime emits this entire string for free.
    ///
    /// Returns `None` if `name` is not registered. The first argument key is
    /// jumped ONLY when it is provably the sole legal opening key: the schema is
    /// closed (`additionalProperties:false`) AND its single required key is its
    /// only declared property. If the schema has optional properties (so the
    /// object could legally begin with a different key) the scaffold stops at the
    /// object opener `{`, preserving losslessness (never force a non-forced token).
    pub fn scaffold_for(&self, name: &str, sole_key: bool) -> Option<String> {
        let schema = self.schemas.iter().find(|s| s.name == name)?;
        let mut out = format!("{{\"name\": \"{name}\", \"arguments\": {{");
        let key_is_forced = sole_key
            && schema.closed
            && schema.required_keys.len() == 1
            && schema.declarable_keys == 1;
        if key_is_forced {
            out.push_str(&format!("\"{}\": ", schema.required_keys[0]));
        }
        Some(out)
    }

    /// Validity gate (Phase 2): is `(name, args)` a legal call under the grammar.
    /// Used both as the constrained-decode invariant and as a cheap post-hoc check.
    pub fn is_valid_call(&self, name: &str, args: &Value) -> Result<(), String> {
        let Some(schema) = self.schemas.iter().find(|s| s.name == name) else {
            return Err(format!("unknown tool \"{name}\""));
        };
        let Some(obj) = args.as_object() else {
            return Err("arguments must be a JSON object".to_string());
        };
        for key in &schema.required_keys {
            if !obj.contains_key(key) {
                return Err(format!("missing required argument \"{key}\""));
            }
        }
        Ok(())
    }

    /// The fraction of a fully-rendered canonical call for `name` that is grammar
    /// -forced scaffolding (envelope + name + punctuation), given the rendered
    /// arguments JSON. This is the "how much did jump-forward save" measure the
    /// runtime reports; it is a lower bound (prompt-lookup saves more on top).
    pub fn forced_fraction(&self, name: &str, arguments_json: &str) -> Option<f64> {
        let scaffold = self.scaffold_for(name, false)?;
        let total = scaffold.len() + arguments_json.len() + self.envelope_suffix().len();
        if total == 0 {
            return Some(0.0);
        }
        let forced = scaffold.len() + self.envelope_suffix().len();
        Some(forced as f64 / total as f64)
    }
}

/// Longest common prefix of a non-empty slice of strings (byte-safe: only cuts on
/// a char boundary because all inputs are `&str` and we compare whole chars).
fn longest_common_prefix(items: &[&str]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut prefix = *first;
    for s in &items[1..] {
        let mut end = 0;
        for ((i, a), b) in prefix.char_indices().zip(s.chars()) {
            if a == b {
                end = i + a.len_utf8();
            } else {
                break;
            }
        }
        prefix = &prefix[..end];
        if prefix.is_empty() {
            break;
        }
    }
    prefix.to_string()
}

// ---------------------------------------------------------------------------
// prompt-lookup drafter (copied argument values)
// ---------------------------------------------------------------------------

/// A training-free n-gram / prompt-lookup drafter. It proposes a continuation for
/// the generated text by finding the tail of what has been generated inside a
/// haystack (the prompt / a file just read) and copying what follows there.
#[derive(Debug, Clone)]
pub struct PromptLookup {
    /// Longest suffix (in chars) to try to match first.
    pub max_ngram: usize,
    /// Shortest suffix to accept a match on.
    pub min_ngram: usize,
}

impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            max_ngram: 32,
            min_ngram: 3,
        }
    }
}

impl PromptLookup {
    pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
        Self {
            max_ngram: max_ngram.max(min_ngram),
            min_ngram: min_ngram.max(1),
        }
    }

    /// Draft up to `max_draft` characters continuing `generated`, by matching the
    /// longest suffix of `generated` (between `min_ngram` and `max_ngram`) that
    /// occurs in `haystack` and returning the text that follows it there. Returns
    /// `None` when no suffix of the allowed lengths matches with any follow-on.
    ///
    /// The runtime feeds the drafted chars (tokenized) to the target verifier, so
    /// a wrong guess costs nothing but the verify it would have done anyway.
    pub fn draft(&self, generated: &str, haystack: &str, max_draft: usize) -> Option<String> {
        if max_draft == 0 || generated.is_empty() || haystack.is_empty() {
            return None;
        }
        let gen_chars: Vec<char> = generated.chars().collect();
        let hi = self.max_ngram.min(gen_chars.len());
        for k in (self.min_ngram..=hi).rev() {
            let suffix: String = gen_chars[gen_chars.len() - k..].iter().collect();
            // First occurrence that has following characters.
            let mut search_from = 0;
            while let Some(rel) = haystack[search_from..].find(&suffix) {
                let end = search_from + rel + suffix.len();
                if end < haystack.len() {
                    let draft: String = haystack[end..].chars().take(max_draft).collect();
                    if !draft.is_empty() {
                        return Some(draft);
                    }
                }
                // advance past this occurrence to look for a later one with follow-on
                search_from = search_from + rel + 1;
                if search_from >= haystack.len() {
                    break;
                }
            }
        }
        None
    }
}

/// How many leading characters of `draft` the target actually accepts, given the
/// ground-truth continuation `truth` the target would have produced. This is the
/// lossless-verify accounting: the accepted count is the length of the common
/// prefix, and the first mismatch is where real decoding resumes. Pure, so the
/// governor and tests can reason about acceptance without a model.
pub fn accepted_prefix_len(draft: &str, truth: &str) -> usize {
    draft
        .chars()
        .zip(truth.chars())
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn grammar() -> ToolCallGrammar {
        ToolCallGrammar::new(vec![
            ToolSchema::new("fs.read", vec!["path".into()]),
            ToolSchema::new("fs.list", vec!["path".into()]),
            ToolSchema::new("git.status", vec![]),
            ToolSchema::new("shell.run", vec!["argv".into()]),
        ])
    }

    #[test]
    fn envelope_prefix_is_forced_opening() {
        assert_eq!(grammar().envelope_prefix(), "{\"name\": \"");
    }

    #[test]
    fn name_jump_shares_prefix_then_resolves() {
        let g = grammar();
        // Empty typed: fs.* and git.* and shell.* share nothing -> no forced chars.
        assert_eq!(g.name_jump("").forced, "");
        // "fs." is shared by fs.read and fs.list: typing "f" forces "s." (the LCP).
        let j = g.name_jump("f");
        assert_eq!(j.forced, "s.");
        assert_eq!(j.resolved, None);
        // "fs.r" uniquely identifies fs.read: forced completes it, resolved set.
        let j = g.name_jump("fs.r");
        assert_eq!(j.forced, "ead");
        assert_eq!(j.resolved.as_deref(), Some("fs.read"));
        // "git." uniquely identifies git.status.
        assert_eq!(g.name_jump("git.").resolved.as_deref(), Some("git.status"));
    }

    #[test]
    fn name_jump_flags_dead_branch() {
        let j = grammar().name_jump("nope");
        assert!(j.dead);
        assert_eq!(j.resolved, None);
    }

    #[test]
    fn scaffold_forces_sole_key_only_when_closed_single_prop() {
        // A closed schema whose ONLY declared property IS the single required key:
        // "path" is provably the only legal first key, so jumping it is lossless.
        let g = ToolCallGrammar::new(vec![ToolSchema::from_input_schema(
            "x.single",
            &json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }),
        )]);
        assert_eq!(
            g.scaffold_for("x.single", true).unwrap(),
            "{\"name\": \"x.single\", \"arguments\": {\"path\": "
        );
        // sole_key=false always stops at the object opener.
        assert_eq!(
            g.scaffold_for("x.single", false).unwrap(),
            "{\"name\": \"x.single\", \"arguments\": {"
        );
        assert_eq!(g.scaffold_for("made.up", true), None);
    }

    #[test]
    fn scaffold_does_not_force_key_when_optional_props_exist() {
        // fs.read's REAL schema: required [path] but optional range/encoding, so the
        // arguments object may legally begin with a non-required key. Forcing "path"
        // would emit a token that was not the only legal continuation (a losslessness
        // violation), so the scaffold must stop at the object opener.
        let g = ToolCallGrammar::new(vec![ToolSchema::from_input_schema(
            "fs.read",
            &json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "range": { "type": "array" },
                    "encoding": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        )]);
        assert_eq!(
            g.scaffold_for("fs.read", true).unwrap(),
            "{\"name\": \"fs.read\", \"arguments\": {"
        );
    }

    #[test]
    fn scaffold_does_not_force_key_when_multiple_required() {
        let g = ToolCallGrammar::new(vec![ToolSchema::new(
            "x.multi",
            vec!["a".into(), "b".into()],
        )]);
        // Two required keys: order is not forced, so only the object opener is emitted.
        assert_eq!(
            g.scaffold_for("x.multi", true).unwrap(),
            "{\"name\": \"x.multi\", \"arguments\": {"
        );
    }

    #[test]
    fn validity_gate_checks_name_and_required_keys() {
        let g = grammar();
        assert!(g.is_valid_call("fs.read", &json!({ "path": "a" })).is_ok());
        assert!(g.is_valid_call("fs.read", &json!({})).is_err());
        assert!(g.is_valid_call("made.up", &json!({})).is_err());
        assert!(g.is_valid_call("fs.read", &json!("not object")).is_err());
        assert!(g.is_valid_call("git.status", &json!({})).is_ok());
    }

    #[test]
    fn forced_fraction_is_a_real_lower_bound() {
        let g = grammar();
        // git.status with empty args: almost all of it is forced scaffolding.
        let frac = g.forced_fraction("git.status", "{}").unwrap();
        assert!(frac > 0.9, "expected mostly-forced, got {frac}");
        // A call with a long argument value has a lower forced fraction.
        let low = g
            .forced_fraction("shell.run", "{\"argv\": [\"a very long command here\"]}")
            .unwrap();
        assert!(low < frac);
    }

    #[test]
    fn schema_from_input_schema_reads_required() {
        let s = ToolSchema::from_input_schema(
            "fs.read",
            &json!({ "type": "object", "required": ["path"] }),
        );
        assert_eq!(s.required_keys, vec!["path".to_string()]);
    }

    #[test]
    fn prompt_lookup_copies_run_from_context() {
        let lookup = PromptLookup::default();
        // The model has emitted a path prefix that appears in the file it read.
        let haystack = "files: src/main.rs, src/lib.rs, README.md";
        let generated = "open src/li";
        let draft = lookup.draft(generated, haystack, 6).unwrap();
        assert_eq!(draft, "b.rs, ");
    }

    #[test]
    fn prompt_lookup_returns_none_without_match() {
        let lookup = PromptLookup::default();
        assert!(lookup.draft("zzz qqq", "nothing alike here", 8).is_none());
    }

    #[test]
    fn accepted_prefix_len_is_common_prefix() {
        assert_eq!(accepted_prefix_len("src/lib.rs", "src/lib.rs"), 10);
        assert_eq!(accepted_prefix_len("src/lib.rs", "src/main"), 4);
        assert_eq!(accepted_prefix_len("abc", "xyz"), 0);
    }

    #[test]
    fn longest_common_prefix_basic() {
        assert_eq!(longest_common_prefix(&["fs.read", "fs.list"]), "fs.");
        assert_eq!(longest_common_prefix(&["a", "b"]), "");
        assert_eq!(longest_common_prefix(&["only"]), "only");
    }
}
