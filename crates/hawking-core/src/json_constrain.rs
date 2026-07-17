//! JSON-mode constrained sampling — token-level logit masking.
//!
//! Maintains a lightweight state machine that tracks which bytes are valid
//! next in a well-formed JSON value. Before sampling each token, masks out
//! (sets to -inf) any token whose decoded text cannot continue the current
//! JSON state.
//!
//! Usage:
//!   let constraint = JsonConstraint::new();
//!   // before each sample call:
//!   constraint.mask_logits(&vocab_index, &mut logits);
//!   let tok = sampler.sample(&mut logits, &params);
//!   let text = tokenizer.decode_one(tok)?;
//!   constraint.advance(&text);

const NEG_INF: f32 = f32::NEG_INFINITY;

// ─── Vocabulary index ─────────────────────────────────────────────────────────

/// One-time per-model lookup table mapping token_id → decoded text.
/// Built at serve startup (or lazily on first json-mode request).
pub struct JsonVocabIndex {
    pub token_text: Vec<String>,
}

impl JsonVocabIndex {
    /// Build the index. `decode_one` is called for every token id in 0..vocab_size.
    pub fn build(vocab_size: usize, decode_one: impl Fn(u32) -> String) -> Self {
        let token_text: Vec<String> = (0..vocab_size as u32).map(decode_one).collect();
        Self { token_text }
    }

    pub fn text(&self, id: u32) -> &str {
        self.token_text
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    pub fn len(&self) -> usize {
        self.token_text.len()
    }
}

// ─── JSON state machine ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum JsonState {
    Start,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectAfterValue,
    ArrayValue,
    ArrayAfterValue,
    InString { escape: bool, is_key: bool },
    InNumber,
    InKeyword { remaining: u8 },
    Done,
}

#[derive(Debug, Clone)]
pub struct JsonConstraint {
    pub state: JsonState,
    depth: Vec<u8>, // b'O' = object, b'A' = array
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonConstraint {
    pub fn new() -> Self {
        Self {
            state: JsonState::Start,
            depth: Vec::new(),
        }
    }

    /// Update state given the decoded text of the last emitted token.
    pub fn advance(&mut self, text: &str) {
        for ch in text.chars() {
            self.advance_char(ch);
        }
    }

    fn advance_char(&mut self, ch: char) {
        use JsonState::*;
        self.state = match &self.state {
            Start => match ch {
                '{' => {
                    self.depth.push(b'O');
                    ObjectKey
                }
                '[' => {
                    self.depth.push(b'A');
                    ArrayValue
                }
                ' ' | '\n' | '\t' | '\r' => Start,
                _ => Start,
            },
            ObjectKey => match ch {
                '"' => InString {
                    escape: false,
                    is_key: true,
                },
                '}' => {
                    self.depth.pop();
                    self.pop_or_done()
                }
                ' ' | '\n' | '\t' => ObjectKey,
                _ => ObjectKey,
            },
            InString { escape, is_key } => {
                if *escape {
                    InString {
                        escape: false,
                        is_key: *is_key,
                    }
                } else if ch == '\\' {
                    InString {
                        escape: true,
                        is_key: *is_key,
                    }
                } else if ch == '"' {
                    if *is_key {
                        ObjectColon
                    } else {
                        match self.depth.last() {
                            Some(&b'O') => ObjectAfterValue,
                            Some(&b'A') => ArrayAfterValue,
                            _ => Done,
                        }
                    }
                } else {
                    InString {
                        escape: false,
                        is_key: *is_key,
                    }
                }
            }
            ObjectColon => {
                if ch == ':' {
                    ObjectValue
                } else {
                    ObjectColon
                }
            }
            ObjectValue => self.start_value(ch),
            ObjectAfterValue => match ch {
                ',' => ObjectKey,
                '}' => {
                    self.depth.pop();
                    self.pop_or_done()
                }
                ' ' | '\n' | '\t' => ObjectAfterValue,
                _ => ObjectAfterValue,
            },
            ArrayValue => self.start_value(ch),
            ArrayAfterValue => match ch {
                ',' => ArrayValue,
                ']' => {
                    self.depth.pop();
                    self.pop_or_done()
                }
                ' ' | '\n' | '\t' => ArrayAfterValue,
                _ => ArrayAfterValue,
            },
            InNumber => match ch {
                '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => InNumber,
                ',' | '}' | ']' | ' ' | '\n' | '\t' => {
                    self.state = match self.depth.last() {
                        Some(&b'O') => ObjectAfterValue,
                        Some(&b'A') => ArrayAfterValue,
                        _ => Done,
                    };
                    self.advance_char(ch);
                    return;
                }
                _ => self.state.clone(),
            },
            InKeyword { remaining } => {
                let r = remaining.saturating_sub(1);
                if r == 0 {
                    match self.depth.last() {
                        Some(&b'O') => ObjectAfterValue,
                        Some(&b'A') => ArrayAfterValue,
                        _ => Done,
                    }
                } else {
                    InKeyword { remaining: r }
                }
            }
            Done => Done,
        };
    }

    fn start_value(&mut self, ch: char) -> JsonState {
        use JsonState::*;
        match ch {
            '"' => InString {
                escape: false,
                is_key: false,
            },
            '{' => {
                self.depth.push(b'O');
                ObjectKey
            }
            '[' => {
                self.depth.push(b'A');
                ArrayValue
            }
            '0'..='9' | '-' => InNumber,
            't' => InKeyword { remaining: 3 },
            'f' => InKeyword { remaining: 4 },
            'n' => InKeyword { remaining: 3 },
            ' ' | '\n' | '\t' => match self.depth.last() {
                Some(&b'O') => ObjectValue,
                _ => ArrayValue,
            },
            _ => match self.depth.last() {
                Some(&b'O') => ObjectValue,
                _ => ArrayValue,
            },
        }
    }

    fn pop_or_done(&self) -> JsonState {
        use JsonState::*;
        match self.depth.last() {
            Some(&b'O') => ObjectAfterValue,
            Some(&b'A') => ArrayAfterValue,
            None => Done,
            _ => Done,
        }
    }

    /// Returns true when the top-level JSON value is fully closed.
    pub fn is_done(&self) -> bool {
        self.state == JsonState::Done
    }

    /// Mask out tokens whose text cannot legally continue the current JSON state.
    /// Sets logit to -inf for any token that starts with an invalid byte.
    ///
    /// This is a prefix check: a token is allowed if its first character (and
    /// any subsequent characters it deterministically commits to) is consistent
    /// with the valid next-byte set for the current state.
    pub fn mask_logits(&self, vocab: &JsonVocabIndex, logits: &mut [f32]) {
        let valid = self.valid_first_bytes();
        for (id, logit) in logits.iter_mut().enumerate() {
            if *logit == NEG_INF {
                continue;
            }
            let text = vocab.text(id as u32);
            if text.is_empty() {
                // Empty token (BOS, padding) — allow so generation doesn't deadlock.
                continue;
            }
            let first = text.chars().next().unwrap();
            if !self.byte_allowed(first, &valid) {
                *logit = NEG_INF;
            }
        }
    }

    fn byte_allowed(&self, ch: char, valid: &ValidFirstBytes) -> bool {
        match valid {
            ValidFirstBytes::Any => true,
            ValidFirstBytes::Set(set) => set.contains(&ch),
            ValidFirstBytes::InString => {
                // Any char except unescaped '"' is allowed inside a string.
                // We allow '"' here too — the state machine will close the string.
                true
            }
            ValidFirstBytes::Done => ch.is_whitespace() || ch == '\n',
        }
    }

    fn valid_first_bytes(&self) -> ValidFirstBytes {
        use JsonState::*;
        match &self.state {
            Start => ValidFirstBytes::Set(vec!['{', '[', ' ', '\n', '\t']),
            ObjectKey => ValidFirstBytes::Set(vec!['"', '}', ' ', '\n', '\t']),
            ObjectColon => ValidFirstBytes::Set(vec![':', ' ', '\n', '\t']),
            ObjectValue | ArrayValue => ValidFirstBytes::Set(vec![
                '"', '{', '[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 't', 'f',
                'n', ' ', '\n', '\t',
            ]),
            ObjectAfterValue => ValidFirstBytes::Set(vec![',', '}', ' ', '\n', '\t']),
            ArrayAfterValue => ValidFirstBytes::Set(vec![',', ']', ' ', '\n', '\t']),
            InString { .. } => ValidFirstBytes::InString,
            InNumber => ValidFirstBytes::Set(vec![
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E', '+', '-', ',',
                '}', ']', ' ', '\n',
            ]),
            InKeyword { remaining } => {
                // For true/false/null we allow any lowercase letter and continuation.
                let _ = remaining;
                ValidFirstBytes::Any
            }
            Done => ValidFirstBytes::Done,
        }
    }
}

enum ValidFirstBytes {
    Any,
    Set(Vec<char>),
    InString,
    Done,
}

/// A structured output constraint a request can carry into the runtime (the
/// W-F4-1 foundation). The engine today has only a binary `json_mode`; this is
/// the core-owned grammar TYPE the runtime-grammar work builds on (core cannot
/// reach hawking-orch's shell-side `GrammarSpec`). `validate` is the post-hoc
/// gate; per-token MASK enforcement of `required_keys` / `Choices` during decode
/// is the deferred runtime FSM (hawking-orch grammar.rs marks it
/// "RUNTIME-SIDE — LATER"), as is threading a `grammar` field through the 44
/// `GenerateRequest` construction sites.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarConstraint {
    /// Any valid JSON object, optionally requiring these top-level keys.
    JsonObject { required_keys: Vec<String> },
    /// Output must be exactly one of these choices (classifier label / enum).
    Choices(Vec<String>),
}

impl GrammarConstraint {
    /// Post-hoc validation gate: does a completed `output` satisfy the constraint?
    pub fn validate(&self, output: &str) -> bool {
        match self {
            GrammarConstraint::JsonObject { required_keys } => {
                match serde_json::from_str::<serde_json::Value>(output.trim()) {
                    Ok(serde_json::Value::Object(map)) => {
                        required_keys.iter().all(|k| map.contains_key(k.as_str()))
                    }
                    _ => false,
                }
            }
            GrammarConstraint::Choices(choices) => {
                let t = output.trim();
                choices.iter().any(|c| c.as_str() == t)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object() {
        let mut c = JsonConstraint::new();
        c.advance("{}");
        assert!(c.is_done());
    }

    #[test]
    fn simple_kv() {
        let mut c = JsonConstraint::new();
        c.advance(r#"{"key": "value"}"#);
        assert!(c.is_done());
    }

    #[test]
    fn nested() {
        let mut c = JsonConstraint::new();
        c.advance(r#"{"a": {"b": 1}}"#);
        assert!(c.is_done());
    }

    #[test]
    fn array() {
        let mut c = JsonConstraint::new();
        c.advance("[1, 2, 3]");
        assert!(c.is_done());
    }

    #[test]
    fn not_done_mid_string() {
        let mut c = JsonConstraint::new();
        c.advance(r#"{"k": "#);
        assert!(!c.is_done());
        assert_eq!(c.state, JsonState::ObjectValue);
    }

    #[test]
    fn grammar_constraint_validates_json_object_required_keys() {
        let c = GrammarConstraint::JsonObject {
            required_keys: vec!["tool".into(), "args".into()],
        };
        assert!(c.validate(r#"{"tool":"grep","args":{}}"#));
        assert!(!c.validate(r#"{"tool":"grep"}"#), "missing required key");
        assert!(!c.validate("not json"));
        assert!(!c.validate("[1,2,3]"), "array is not an object");
        let any = GrammarConstraint::JsonObject {
            required_keys: vec![],
        };
        assert!(any.validate(r#"{"x":1}"#));
    }

    #[test]
    fn grammar_constraint_validates_choices() {
        let c = GrammarConstraint::Choices(vec!["yes".into(), "no".into()]);
        assert!(c.validate("yes"));
        assert!(c.validate("  no  "), "trims whitespace");
        assert!(!c.validate("maybe"));
    }
}
