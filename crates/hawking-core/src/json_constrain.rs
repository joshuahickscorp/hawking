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
    /// Tokenizer vocabulary size. Logit ids at or beyond this are padding
    /// and are always masked. Defaults to `token_text.len()` from [`Self::build`].
    known_len: usize,
}

impl JsonVocabIndex {
    /// Build the index. `decode_one` is called for every token id in 0..vocab_size.
    ///
    /// `known_len` defaults to `vocab_size`. Call [`Self::with_known_len`] when
    /// the logits buffer is wider than the tokenizer vocabulary.
    pub fn build(vocab_size: usize, decode_one: impl Fn(u32) -> String) -> Self {
        let token_text: Vec<String> = (0..vocab_size as u32).map(decode_one).collect();
        Self {
            token_text,
            known_len: vocab_size,
        }
    }

    /// Restrict which ids are considered in-vocabulary.
    ///
    /// Ids at or beyond `known_len` are always masked, even if they decode to
    /// the empty string. Empty-text allowance (BOS/EOS/`<|im_end|>`) applies
    /// only below this cutoff.
    pub fn with_known_len(mut self, known_len: usize) -> Self {
        self.known_len = known_len;
        self
    }

    pub fn known_len(&self) -> usize {
        self.known_len
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

/// Deterministic greedy argmax matching `sample_argmax_f32` in
/// `crates/hawking-core/shaders/sample.metal`.
///
/// The kernel reduces with `if (vb > va || (vb == va && ib < ia))` (line 69):
/// a strictly greater value wins; an exact tie keeps the lower index. The
/// per-thread scan uses `if (v > local_v)` walking `i = tid, tid+tg, ...`,
/// which is the same rule (equal values never replace an earlier index).
pub fn argmax_f32_metal_tiebreak(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
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
        let known_len = vocab.known_len;
        for (id, logit) in logits.iter_mut().enumerate() {
            if *logit == NEG_INF {
                continue;
            }
            if id >= known_len {
                *logit = NEG_INF;
                continue;
            }
            let text = vocab.text(id as u32);
            if text.is_empty() {
                // Empty token below known_len (BOS, EOS, <|im_end|>). Allowing
                // these UNCONDITIONALLY let the model end the reply in the
                // middle of the object the constraint exists to complete.
                // Measured: three attempts, stop_reason "eos", finish_reason
                // "stop", 868 to 915 completion tokens against a budget of
                // 5874 -- the model simply stopped inside a "new_lines" array
                // and the host reported "Expecting ',' delimiter" at the last
                // character of the reply, while every call recorded
                // grammar_enforced = true.
                //
                // Terminate only once the object is CLOSED. If the model will
                // not close it, running to the token budget is the honest
                // outcome: stop_reason "budget" says what happened, where a
                // permitted EOS produced a truncated reply indistinguishable
                // from a malformed one.
                if self.is_done() {
                    continue;
                }
                *logit = NEG_INF;
                continue;
            }
            // Check the WHOLE token, not just its first character. Tokens are
            // BPE pieces of several characters, so one can begin legally and
            // break JSON in its tail -- `",` after a value, or text carrying a
            // raw newline. The mask reported grammar_enforced=true while the
            // reply still failed to parse:
            //
            //   Expecting ',' delimiter: line 9 column 143
            //
            // The first-character test is kept as a cheap reject: most tokens
            // fail it outright in a constrained state, and only survivors pay
            // for the full simulation.
            let first = text.chars().next().unwrap();
            if !self.byte_allowed(first, &valid) || !self.token_allowed(text) {
                *logit = NEG_INF;
            }
        }
    }

    /// Would every character of `text` be legal, in order, from here?
    ///
    /// A token is atomic: the model emits all of it or none of it, so a token
    /// whose tail is illegal must be refused whole.
    fn token_allowed(&self, text: &str) -> bool {
        let mut probe = self.clone();
        for ch in text.chars() {
            let valid = probe.valid_first_bytes();
            if !probe.byte_allowed(ch, &valid) {
                return false;
            }
            probe.advance_char(ch);
        }
        true
    }

    fn byte_allowed(&self, ch: char, valid: &ValidFirstBytes) -> bool {
        match valid {
            ValidFirstBytes::Any => true,
            ValidFirstBytes::Set(set) => set.contains(&ch),
            ValidFirstBytes::InString => {
                // A raw control character is NOT legal inside a JSON string.
                // RFC 8259 requires U+0000..=U+001F to be escaped, so allowing
                // "any char except an unescaped quote" let the model emit a
                // literal newline or tab in a string value and produce a reply
                // that could not parse:
                //
                //   the reply is NOT valid JSON -- Invalid control character
                //   at: line 67 column 23
                //
                // That is the one failure this mask exists to make impossible,
                // and it happened while the mask was running. Embedding source
                // code in a string value is the common case, and it is exactly
                // the case that emits raw newlines.
                //
                // '"' stays allowed: the state machine closes the string on it.
                (ch as u32) >= 0x20
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
    /// A raw control character inside a string is not legal JSON.
    ///
    /// The mask allowed "any char except an unescaped quote", so a model
    /// embedding source code in a string value emitted a literal newline and
    /// produced a reply that could not parse:
    ///
    ///   the reply is NOT valid JSON -- Invalid control character at:
    ///   line 67 column 23
    ///
    /// That happened WHILE the mask was running: the one failure it exists to
    /// prevent.
    #[test]
    fn raw_control_characters_are_refused_inside_a_string() {
        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"code");
        let valid = c.valid_first_bytes();
        assert!(!c.byte_allowed('\n', &valid), "a raw newline must be refused");
        assert!(!c.byte_allowed('\t', &valid), "a raw tab must be refused");
        assert!(!c.byte_allowed('\u{0}', &valid), "a raw NUL must be refused");
    }

    /// Negative control: escaping and ordinary text must still be accepted, or
    /// the mask would make it impossible to write code into a string at all.
    #[test]
    fn escapes_and_ordinary_text_are_still_allowed_inside_a_string() {
        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"code");
        let valid = c.valid_first_bytes();
        for ch in ['x', ' ', '\\', '"', '{', 'é'] {
            assert!(c.byte_allowed(ch, &valid), "{ch:?} must stay allowed");
        }
    }

    /// A token that starts legally and breaks JSON in its tail must be
    /// refused whole. This is what made grammar_enforced=true coexist with
    /// "Expecting ',' delimiter" in a live receipt.
    #[test]
    fn a_token_whose_tail_breaks_json_is_refused() {
        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"v\"");
        // After a closed string value the only legal next characters are , } or
        // whitespace. A BPE piece like ":x" starts illegally; one like ",,"
        // starts LEGALLY and is still invalid.
        assert!(!c.token_allowed(",,"), "a legal first char must not admit an illegal tail");
        assert!(c.token_allowed(","), "the legal single character must stay allowed");
    }

    /// Inside a string, a token carrying a raw newline must be refused even
    /// though its first character is ordinary text.
    #[test]
    fn a_string_token_carrying_a_raw_newline_is_refused() {
        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"code");
        assert!(!c.token_allowed("x\ny"), "a raw newline in the tail must be refused");
        assert!(c.token_allowed("xy"), "ordinary text must stay allowed");
        assert!(c.token_allowed("x\\ny"), "an ESCAPED newline must stay allowed");
    }

    /// Through mask_logits, not the helper.
    ///
    /// The first version of these tests called `token_allowed` directly, so
    /// they all passed with the whole-token check removed from `mask_logits` --
    /// the helper was never the thing that failed. Same trap as testing a
    /// prompt sanitizer instead of the assembled packet.
    #[test]
    fn mask_logits_refuses_a_token_whose_tail_breaks_json() {
        // A BPE piece that starts legally and is invalid as a whole: after a
        // closed string value, "," is legal and ",," is not.
        const TAIL_BREAKS: usize = 2;
        let texts = ["", ",", ",,", "}"];
        let vocab = JsonVocabIndex::build(texts.len(), |id| texts[id as usize].to_string());

        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"v\"");

        let mut logits = vec![1.0f32; vocab.len()];
        logits[TAIL_BREAKS] = 100.0;
        c.mask_logits(&vocab, &mut logits);

        assert_eq!(
            logits[TAIL_BREAKS], NEG_INF,
            "mask_logits admitted a token whose tail breaks JSON",
        );
        assert_ne!(logits[1], NEG_INF, "the legal single comma must survive");
    }

    /// Same, inside a string: a token carrying a raw newline in its tail.
    #[test]
    fn mask_logits_refuses_a_string_token_carrying_a_raw_newline() {
        const RAW_NEWLINE: usize = 1;
        let texts = ["", "x\ny", "xy"];
        let vocab = JsonVocabIndex::build(texts.len(), |id| texts[id as usize].to_string());

        let mut c = JsonConstraint::new();
        c.advance("{\"a\": \"code");

        let mut logits = vec![1.0f32; vocab.len()];
        logits[RAW_NEWLINE] = 100.0;
        c.mask_logits(&vocab, &mut logits);

        assert_eq!(logits[RAW_NEWLINE], NEG_INF, "raw newline admitted inside a string");
        assert_ne!(logits[2], NEG_INF, "ordinary text must survive");
    }

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

    /// Host argmax must pick the same winner as `sample_argmax_f32`.
    ///
    /// Derived from `crates/hawking-core/shaders/sample.metal` line 69:
    /// `if (vb > va || (vb == va && ib < ia)) { shmem_v[tid] = vb; shmem_i[tid] = ib; }`
    /// Strictly greater wins; an exact tie keeps the lower index.
    #[test]
    fn host_argmax_matches_metal_lower_index_on_tie() {
        let logits = [1.5f32, 3.0, 3.0, 2.0];
        assert_eq!(
            argmax_f32_metal_tiebreak(&logits),
            1,
            "exact tie at 3.0 must keep the lower index (shader line 69)"
        );
        let unique = [0.0f32, 1.0, 4.0, 3.0];
        assert_eq!(argmax_f32_metal_tiebreak(&unique), 2);
        let leading = [5.0f32, 5.0, 5.0];
        assert_eq!(argmax_f32_metal_tiebreak(&leading), 0);
        assert_eq!(argmax_f32_metal_tiebreak(&[]), 0);
    }

    /// Why `generate_constrained` guards before calling this.
    ///
    /// An all-masked vector has no finite entry, so the scan never fires and
    /// this returns id 0 -- a real token. Without the caller's
    /// `logits.iter().any(is_finite)` check the resident would emit token 0 for
    /// the rest of the budget while still reporting `grammar_enforced: true`:
    /// a silent wrong answer wearing an enforcement claim. If this ever stops
    /// returning 0 for an all-masked vector, revisit that guard rather than
    /// assuming it became unnecessary.
    #[test]
    fn a_fully_masked_vector_argmaxes_to_a_real_token_id() {
        let all_masked = [NEG_INF; 8];
        assert_eq!(
            argmax_f32_metal_tiebreak(&all_masked),
            0,
            "the caller must detect this case before sampling"
        );
        let one_survivor = [NEG_INF, NEG_INF, -12.5, NEG_INF];
        assert_eq!(argmax_f32_metal_tiebreak(&one_survivor), 2);
    }

    #[test]
    fn eos_becomes_emittable_once_the_object_is_closed() {
        // The other half of the contract. Masking EOS forever would make a
        // completed reply impossible to end.
        const EOS: usize = 0;
        let vocab = JsonVocabIndex::build(4, |id| match id {
            0 => String::new(),
            1 => "{".into(),
            2 => "}".into(),
            3 => "true".into(),
            _ => String::new(),
        });
        let mut c = JsonConstraint::new();
        c.advance("{}");
        assert!(c.is_done(), "fixture must actually close the object");
        let mut logits = vec![0.0f32; 8];
        c.mask_logits(&vocab, &mut logits);
        assert_ne!(
            logits[EOS], NEG_INF,
            "a closed object must be allowed to terminate"
        );
    }

    #[test]
    fn padding_ids_at_or_beyond_known_len_are_masked_eos_is_not() {
        // id 0 is EOS (empty text, below known_len). ids 4.. are padding
        // in a logits buffer wider than the tokenizer vocabulary.
        const EOS: usize = 0;
        let vocab = JsonVocabIndex::build(4, |id| match id {
            0 => String::new(),
            1 => "{".into(),
            2 => "}".into(),
            3 => "true".into(),
            _ => String::new(),
        });
        assert_eq!(vocab.known_len(), 4);
        let c = JsonConstraint::new();
        let mut logits = vec![0.0f32; 8];
        logits[EOS] = 1.0;
        logits[1] = 0.5;
        logits[7] = 99.0;
        c.mask_logits(&vocab, &mut logits);
        assert_eq!(
            logits[7], NEG_INF,
            "id 7 is at/beyond known_len and must be masked"
        );
        assert_eq!(
            logits[4], NEG_INF,
            "id 4 is at known_len and must be masked"
        );
        assert_eq!(
            logits[EOS], NEG_INF,
            "EOS must be masked while the object is still OPEN -- allowing it \
             unconditionally let three consecutive replies stop inside a JSON \
             array with grammar_enforced reported true"
        );
        let shrunk = vocab.with_known_len(2);
        let mut logits = vec![5.0f32; 4];
        c.mask_logits(&shrunk, &mut logits);
        assert_eq!(logits[2], NEG_INF);
        assert_eq!(logits[3], NEG_INF);
        // id 0 is EOS and the object is still open, so it is masked here for
        // the same reason as above. id 1 is "{", which is exactly what an
        // unopened object should be allowed to emit.
        assert_eq!(logits[0], NEG_INF);
        assert_ne!(logits[1], NEG_INF);
    }

    /// Load-bearing evidence: a model that always prefers an illegal token
    /// still cannot emit one while the masker is live. The concatenated
    /// output must parse as JSON. Mutation-check this by turning
    /// `mask_logits` into a no-op — the same assertions must then fail.
    #[test]
    fn masked_argmax_cannot_emit_illegal_json_token() {
        const NOPE: usize = 9;
        let texts = [
            "", "{", "}", "[", "]", "\"", ":", ",", "true", "NOPE", "1", " ",
        ];
        let vocab = JsonVocabIndex::build(texts.len(), |id| texts[id as usize].to_string());
        let mut c = JsonConstraint::new();
        let mut out = String::new();
        for _ in 0..16 {
            let mut logits = vec![1.0f32; vocab.len()];
            logits[0] = 0.0;
            logits[NOPE] = 100.0;
            c.mask_logits(&vocab, &mut logits);
            let id = argmax_f32_metal_tiebreak(&logits) as usize;
            assert_ne!(id, NOPE, "masked argmax emitted illegal token NOPE");
            let text = vocab.text(id as u32);
            out.push_str(text);
            c.advance(text);
            if c.is_done() || text.is_empty() {
                break;
            }
        }
        assert!(c.is_done(), "constraint never reached Done; out={out:?}");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out:?}"));
        assert!(
            parsed.is_object() || parsed.is_array(),
            "expected object or array, got {parsed}"
        );
        assert!(!out.contains("NOPE"), "illegal token leaked into {out:?}");
    }
}
