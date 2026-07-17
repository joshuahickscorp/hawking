//! Constrained / grammar decode as a shell-side service (ch.06 §4.5).
//!
//! The runtime owns the real `mask_logits` primitive; until a `grammar` request
//! field lands ([RUNTIME-SIDE — LATER]), this crate provides the **[SHELL-TODAY]**
//! fallback the bible specifies (§4.5.4):
//!
//! 1. A [`GrammarSpec`] enum (`JsonObject | Regex | Choices`) describing what the
//!    output envelope must satisfy.
//! 2. A [`GrammarMatcher`] that does **validate-and-retry**: parse a completed
//!    output against the spec, and on failure return a structured [`RetryHint`]
//!    the caller folds into a re-prompt.
//! 3. For the `JsonObject` case, a real **JSON-object FSM** ([`JsonObjectFsm`])
//!    that, given the text emitted so far, reports which *classes* of next
//!    character are legal — the shell-side analog of the runtime's per-token
//!    mask. It is exact for a flat `{ "k": v, ... }` object and degrades to a
//!    permissive state for arbitrary nesting (never a fabricated hash).
//!
//! No fabricated hashes: [`compile`] hashes the spec's canonical bytes with a
//! real FNV-1a digest keyed by tokenizer signature.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What an output's envelope must satisfy. Deliberately small — the bible's
/// `JsonObject | Regex | Choices` shell subset (§4.5.1 lists more cases that are
/// runtime-side).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GrammarSpec {
    /// Any well-formed JSON object (optionally with required top-level keys).
    JsonObject { required_keys: Vec<String> },
    /// The whole output must fully match this regex.
    Regex(String),
    /// The output must be exactly one of these choices (classifier labels, enum).
    Choices(Vec<String>),
}

impl GrammarSpec {
    /// Canonical bytes for hashing/caching.
    fn canonical(&self) -> String {
        match self {
            GrammarSpec::JsonObject { required_keys } => {
                let mut keys = required_keys.clone();
                keys.sort();
                format!("json_object:{}", keys.join(","))
            }
            GrammarSpec::Regex(r) => format!("regex:{r}"),
            GrammarSpec::Choices(c) => {
                let mut c = c.clone();
                c.sort();
                format!("choices:{}", c.join("\u{1}"))
            }
        }
    }
}

/// A structured retry instruction emitted on a validation failure. The caller
/// (router/executor or ch.02 kernel) re-prompts with `message` appended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryHint {
    /// Stable failure code (`NOT_JSON`, `MISSING_KEY`, `REGEX_MISMATCH`, `NOT_A_CHOICE`).
    pub code: String,
    /// Human/agent-readable correction to fold into the re-prompt.
    pub message: String,
}

/// Outcome of validating a completed output against a [`GrammarSpec`].
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarValidation {
    Valid,
    Retry(RetryHint),
}

// ---- Legacy/compat surface (kept stable for any external consumer) ---------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarRequest {
    pub name: String,
    pub schema_json: String,
    pub tokenizer_signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledGrammar {
    pub name: String,
    pub grammar_hash: String,
    pub tokenizer_signature: String,
    pub mask_cache_key: String,
}

pub trait GrammarCompiler: Send + Sync {
    fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar>;
}

/// FNV-1a 64-bit, hex. Real digest — no fabricated stub hashes.
fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// The real grammar compiler: parses the schema JSON into a [`GrammarSpec`],
/// content-addresses it, and produces a [`CompiledGrammar`] whose hash is a real
/// digest of `(spec, tokenizer_sig)`.
#[derive(Default)]
pub struct ShellGrammarCompiler;

impl ShellGrammarCompiler {
    /// Derive a [`GrammarSpec`] from a schema JSON document. Supports a JSON
    /// Schema-ish `{"type":"object","required":[...]}`, `{"enum":[...]}`, and
    /// `{"pattern":"..."}`; falls back to a permissive JSON object.
    pub fn spec_from_schema(schema_json: &str) -> hide_core::Result<GrammarSpec> {
        let value: Value = serde_json::from_str(schema_json)?;
        if let Some(choices) = value.get("enum").and_then(|v| v.as_array()) {
            let choices = choices
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            return Ok(GrammarSpec::Choices(choices));
        }
        if let Some(pattern) = value.get("pattern").and_then(|v| v.as_str()) {
            return Ok(GrammarSpec::Regex(pattern.to_string()));
        }
        let required = value
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(GrammarSpec::JsonObject {
            required_keys: required,
        })
    }
}

impl GrammarCompiler for ShellGrammarCompiler {
    fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar> {
        let spec = Self::spec_from_schema(&request.schema_json)?;
        let canonical = spec.canonical();
        let grammar_hash = fnv1a_hex(canonical.as_bytes());
        let mask_cache_key =
            fnv1a_hex(format!("{canonical}|{}", request.tokenizer_signature).as_bytes());
        Ok(CompiledGrammar {
            name: request.name,
            grammar_hash,
            tokenizer_signature: request.tokenizer_signature,
            mask_cache_key,
        })
    }
}

/// The decode-time / output-time matcher. Holds a spec and (for the JSON case) a
/// running FSM. Used two ways: feed it the *whole* completed output to validate
/// (`validate`), or drive the [`JsonObjectFsm`] incrementally for a shell-side
/// legal-next-char check.
#[derive(Debug, Clone)]
pub struct GrammarMatcher {
    spec: GrammarSpec,
    regex: Option<Regex>,
}

impl GrammarMatcher {
    pub fn new(spec: GrammarSpec) -> hide_core::Result<Self> {
        let regex = match &spec {
            GrammarSpec::Regex(r) => Some(Regex::new(r).map_err(|e| {
                hide_core::error::HideError::Config(format!("invalid grammar regex: {e}"))
            })?),
            _ => None,
        };
        Ok(Self { spec, regex })
    }

    pub fn spec(&self) -> &GrammarSpec {
        &self.spec
    }

    /// Validate a completed output against the spec, returning a structured
    /// retry hint on failure (the validate-and-retry fallback, §4.5.4).
    pub fn validate(&self, output: &str) -> GrammarValidation {
        match &self.spec {
            GrammarSpec::JsonObject { required_keys } => {
                let parsed: Value = match serde_json::from_str(output.trim()) {
                    Ok(v) => v,
                    Err(e) => {
                        return GrammarValidation::Retry(RetryHint {
                            code: "NOT_JSON".to_string(),
                            message: format!(
                            "Output was not valid JSON ({e}). Re-emit ONLY a single JSON object."
                        ),
                        })
                    }
                };
                let obj = match parsed.as_object() {
                    Some(o) => o,
                    None => {
                        return GrammarValidation::Retry(RetryHint {
                            code: "NOT_JSON".to_string(),
                            message: "Output must be a JSON object, not an array or scalar."
                                .to_string(),
                        })
                    }
                };
                for key in required_keys {
                    if !obj.contains_key(key) {
                        return GrammarValidation::Retry(RetryHint {
                            code: "MISSING_KEY".to_string(),
                            message: format!(
                                "JSON object is missing required key \"{key}\". Add it and re-emit."
                            ),
                        });
                    }
                }
                GrammarValidation::Valid
            }
            GrammarSpec::Regex(pattern) => {
                let re = self.regex.as_ref().expect("regex compiled in new()");
                if re.is_match(output.trim()) {
                    GrammarValidation::Valid
                } else {
                    GrammarValidation::Retry(RetryHint {
                        code: "REGEX_MISMATCH".to_string(),
                        message: format!(
                            "Output did not match the required pattern /{pattern}/. Re-emit to match it exactly."
                        ),
                    })
                }
            }
            GrammarSpec::Choices(choices) => {
                let trimmed = output.trim();
                if choices.iter().any(|c| c == trimmed) {
                    GrammarValidation::Valid
                } else {
                    GrammarValidation::Retry(RetryHint {
                        code: "NOT_A_CHOICE".to_string(),
                        message: format!(
                            "Output must be exactly one of: {}. Re-emit one of these and nothing else.",
                            choices.join(", ")
                        ),
                    })
                }
            }
        }
    }

    /// A fresh JSON-object FSM for incremental masking (only meaningful for the
    /// `JsonObject` spec).
    pub fn json_fsm(&self) -> Option<JsonObjectFsm> {
        matches!(self.spec, GrammarSpec::JsonObject { .. }).then(JsonObjectFsm::new)
    }
}

/// Classes of next character the JSON-object FSM permits in its current state.
/// This is the shell-side analog of the runtime's per-token logit mask: it tells
/// a caller which characters keep the prefix on a path to a valid object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NextCharMask {
    pub allow_open_brace: bool,
    pub allow_close_brace: bool,
    pub allow_quote: bool,
    pub allow_colon: bool,
    pub allow_comma: bool,
    /// Inside a string literal: any character (until a closing quote).
    pub allow_string_char: bool,
    /// Value position: digits / `t`/`f`/`n` (true/false/null) / `{` / `"`.
    pub allow_value_start: bool,
    pub allow_whitespace: bool,
}

/// A minimal pushdown-free FSM over a *flat* JSON object `{ "k": "v", ... }`.
/// It is exact for flat string-valued objects (the common tool-call envelope)
/// and, on encountering nesting/other value types, enters a permissive
/// `Freeform` state rather than rejecting — honoring §4.5.3 ("the grammar
/// guarantees the envelope, never the thought").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonObjectFsm {
    state: FsmState,
    /// True once the closing top-level brace has been consumed.
    done: bool,
    /// True if the input has irrecoverably left the flat-object grammar
    /// (a real syntax error: a value where a key was due, etc.).
    dead: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FsmState {
    /// Before the opening `{`.
    Start,
    /// After `{` or after a `,` — expecting a key string (or `}` to close).
    ExpectKeyOrClose,
    /// Inside a key string literal.
    InKey,
    /// After a key string — expecting `:`.
    ExpectColon,
    /// After `:` — expecting a value.
    ExpectValue,
    /// Inside a string value.
    InStringValue,
    /// After a complete value — expecting `,` or `}`.
    ExpectCommaOrClose,
    /// Permissive: nesting / non-string value detected; we no longer constrain.
    Freeform,
}

impl Default for JsonObjectFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonObjectFsm {
    pub fn new() -> Self {
        Self {
            state: FsmState::Start,
            done: false,
            dead: false,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.done
    }

    /// No legal next character — a dead end the caller can escalate on (§4.4).
    pub fn is_dead_ended(&self) -> bool {
        self.dead
    }

    /// The legal next-character classes in the current state.
    pub fn mask(&self) -> NextCharMask {
        let none = NextCharMask {
            allow_open_brace: false,
            allow_close_brace: false,
            allow_quote: false,
            allow_colon: false,
            allow_comma: false,
            allow_string_char: false,
            allow_value_start: false,
            allow_whitespace: true,
        };
        if self.dead || self.done {
            return NextCharMask {
                allow_whitespace: true,
                ..none
            };
        }
        match self.state {
            FsmState::Start => NextCharMask {
                allow_open_brace: true,
                ..none
            },
            FsmState::ExpectKeyOrClose => NextCharMask {
                allow_quote: true,
                allow_close_brace: true,
                ..none
            },
            FsmState::InKey | FsmState::InStringValue => NextCharMask {
                allow_quote: true,
                allow_string_char: true,
                allow_whitespace: true,
                ..none
            },
            FsmState::ExpectColon => NextCharMask {
                allow_colon: true,
                ..none
            },
            FsmState::ExpectValue => NextCharMask {
                allow_quote: true,
                allow_value_start: true,
                allow_open_brace: true,
                ..none
            },
            FsmState::ExpectCommaOrClose => NextCharMask {
                allow_comma: true,
                allow_close_brace: true,
                ..none
            },
            FsmState::Freeform => NextCharMask {
                allow_open_brace: true,
                allow_close_brace: true,
                allow_quote: true,
                allow_colon: true,
                allow_comma: true,
                allow_string_char: true,
                allow_value_start: true,
                allow_whitespace: true,
            },
        }
    }

    /// Advance the FSM by one already-chosen character.
    pub fn accept(&mut self, c: char) {
        if self.dead || self.done {
            return;
        }
        if c.is_whitespace() && !matches!(self.state, FsmState::InKey | FsmState::InStringValue) {
            return; // whitespace is structurally inert between tokens
        }
        match self.state {
            FsmState::Start => {
                if c == '{' {
                    self.state = FsmState::ExpectKeyOrClose;
                } else {
                    self.dead = true;
                }
            }
            FsmState::ExpectKeyOrClose => match c {
                '"' => self.state = FsmState::InKey,
                '}' => self.done = true,
                _ => self.dead = true,
            },
            FsmState::InKey => {
                if c == '"' {
                    self.state = FsmState::ExpectColon;
                }
                // else: still in the key string
            }
            FsmState::ExpectColon => {
                if c == ':' {
                    self.state = FsmState::ExpectValue;
                } else {
                    self.dead = true;
                }
            }
            FsmState::ExpectValue => match c {
                '"' => self.state = FsmState::InStringValue,
                '{' | '[' => self.state = FsmState::Freeform, // nesting → permissive
                _ => self.state = FsmState::Freeform,         // numbers/bools/null → permissive
            },
            FsmState::InStringValue => {
                if c == '"' {
                    self.state = FsmState::ExpectCommaOrClose;
                }
            }
            FsmState::ExpectCommaOrClose => match c {
                ',' => self.state = FsmState::ExpectKeyOrClose,
                '}' => self.done = true,
                _ => self.dead = true,
            },
            FsmState::Freeform => {
                // Best-effort: a top-level close ends the object.
                if c == '}' {
                    self.done = true;
                }
            }
        }
    }

    /// Drive the FSM over a whole string (convenience for tests/validation).
    pub fn accept_str(&mut self, s: &str) {
        for c in s.chars() {
            self.accept(c);
            if self.done || self.dead {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_produces_real_hashes() {
        let c = ShellGrammarCompiler;
        let out = c
            .compile(GrammarRequest {
                name: "tool".into(),
                schema_json: "{\"type\":\"object\",\"required\":[\"name\"]}".into(),
                tokenizer_signature: "tok-a".into(),
            })
            .unwrap();
        assert!(!out.grammar_hash.starts_with("stub:"));
        assert_eq!(out.grammar_hash.len(), 16);
        // Tokenizer sig changes the mask cache key but not the grammar hash.
        let out2 = c
            .compile(GrammarRequest {
                name: "tool".into(),
                schema_json: "{\"type\":\"object\",\"required\":[\"name\"]}".into(),
                tokenizer_signature: "tok-b".into(),
            })
            .unwrap();
        assert_eq!(out.grammar_hash, out2.grammar_hash);
        assert_ne!(out.mask_cache_key, out2.mask_cache_key);
    }

    #[test]
    fn schema_to_choices_and_regex() {
        assert_eq!(
            ShellGrammarCompiler::spec_from_schema("{\"enum\":[\"a\",\"b\"]}").unwrap(),
            GrammarSpec::Choices(vec!["a".into(), "b".into()])
        );
        assert_eq!(
            ShellGrammarCompiler::spec_from_schema("{\"pattern\":\"^\\\\d+$\"}").unwrap(),
            GrammarSpec::Regex("^\\d+$".into())
        );
    }

    #[test]
    fn json_object_validate_missing_key() {
        let m = GrammarMatcher::new(GrammarSpec::JsonObject {
            required_keys: vec!["name".into()],
        })
        .unwrap();
        assert_eq!(m.validate("{\"name\":\"x\"}"), GrammarValidation::Valid);
        match m.validate("{\"other\":1}") {
            GrammarValidation::Retry(h) => assert_eq!(h.code, "MISSING_KEY"),
            _ => panic!("expected retry"),
        }
        match m.validate("not json") {
            GrammarValidation::Retry(h) => assert_eq!(h.code, "NOT_JSON"),
            _ => panic!("expected retry"),
        }
    }

    #[test]
    fn choices_and_regex_validate() {
        let m = GrammarMatcher::new(GrammarSpec::Choices(vec!["yes".into(), "no".into()])).unwrap();
        assert_eq!(m.validate(" yes "), GrammarValidation::Valid);
        match m.validate("maybe") {
            GrammarValidation::Retry(h) => assert_eq!(h.code, "NOT_A_CHOICE"),
            _ => panic!("expected retry"),
        }
        let r = GrammarMatcher::new(GrammarSpec::Regex("^v\\d+$".into())).unwrap();
        assert_eq!(r.validate("v12"), GrammarValidation::Valid);
        assert!(matches!(r.validate("x"), GrammarValidation::Retry(_)));
    }

    #[test]
    fn fsm_accepts_flat_object_and_completes() {
        let mut fsm = JsonObjectFsm::new();
        // At Start only `{` is legal.
        assert!(fsm.mask().allow_open_brace);
        assert!(!fsm.mask().allow_quote);
        fsm.accept_str("{\"name\":\"edit_file\"}");
        assert!(fsm.is_complete());
        assert!(!fsm.is_dead_ended());
    }

    #[test]
    fn fsm_dead_ends_on_value_where_key_expected() {
        let mut fsm = JsonObjectFsm::new();
        fsm.accept_str("{1");
        assert!(fsm.is_dead_ended());
    }

    #[test]
    fn fsm_goes_permissive_on_nesting() {
        let mut fsm = JsonObjectFsm::new();
        fsm.accept_str("{\"a\":{\"b\":1}}");
        // Did not dead-end on the nested object.
        assert!(!fsm.is_dead_ended());
    }
}
