//! Secret redaction before durability (bible ch.10 §4.8, S6).
//!
//! Replaces the original two-prefix toy with a real detector suite: known-format
//! pattern detectors (AWS access keys, GitHub/GitLab PATs, PEM private-key
//! blocks, JWTs, Slack tokens) plus a generic **Shannon-entropy** detector for
//! high-entropy tokens that no signature catches. On a hit, the span is replaced
//! with `«redacted:detector»` (guillemets U+00AB/U+00BB, lowercase detector
//! name, per bible §4.8) and the location is recorded so the *fact and place* of
//! redaction stays auditable while the secret never enters the log, the chain
//! hash, the blob CAS, or a vector store (§4.2.1 / §4.8).
//!
//! Two surfaces:
//!   * [`Redactor::redact`] — scrub a flat string (shell output, a log line).
//!   * [`Redactor::redact_json`] — scrub every string leaf of a JSON value and
//!     emit the **JSON-pointer paths** (RFC 6901) of redacted leaves, ready to
//!     drop into `Event.redactions` (§4.8). This is the form a `tool.result`
//!     payload goes through before it becomes a durable event.

use hide_core::event::NewEvent;
use hide_core::ids::SessionId;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

/// Opening guillemet of the redaction marker (U+00AB, `«`), per bible §4.8.
pub const MARKER_OPEN: &str = "\u{00AB}";
/// Closing guillemet of the redaction marker (U+00BB, `»`), per bible §4.8.
pub const MARKER_CLOSE: &str = "\u{00BB}";

/// Marker substituted for a redacted span (§4.8): `«redacted:<detector>»` with
/// guillemets (U+00AB / U+00BB) and a **lowercase** detector name, so the audit
/// trail records *why* a span was scrubbed in exactly the form the bible
/// mandates (and the UI renders verbatim, §4.8). Detector names are already
/// lowercase ASCII identifiers; we lowercase defensively for any plugin-supplied
/// detector registered via [`Redactor::with_detector`].
fn marker(detector: &str) -> String {
    format!(
        "{MARKER_OPEN}redacted:{detector}{MARKER_CLOSE}",
        detector = detector.to_ascii_lowercase()
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Redaction {
    /// Detector that matched (e.g. `"aws_access_key"`, `"entropy"`).
    pub pattern_name: String,
    /// The marker that replaced the span.
    pub replacement: String,
    pub occurrences: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionReport {
    pub text: String,
    pub redactions: Vec<Redaction>,
}

impl RedactionReport {
    pub fn is_clean(&self) -> bool {
        self.redactions.is_empty()
    }
}

/// Result of scrubbing a JSON payload (§4.8): the scrubbed value plus the
/// JSON-pointer paths of every redacted string leaf (for `Event.redactions`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRedactionReport {
    pub value: Value,
    /// RFC 6901 JSON-pointer paths of redacted leaves (e.g. `/output/stdout`).
    pub paths: Vec<String>,
    /// Per-detector tallies across the whole document.
    pub redactions: Vec<Redaction>,
}

impl JsonRedactionReport {
    pub fn is_clean(&self) -> bool {
        self.paths.is_empty()
    }

    /// Build the `security.redaction` event the host appends when a payload was
    /// scrubbed before durability (bible §4.8).
    ///
    /// ## Seam
    /// The redactor is *pure*: it never touches the log. The host owns the write
    /// ordering: it (1) runs [`Redactor::redact_json`] on a `tool.result` (or
    /// shell-output, or any pre-durable) payload, (2) sets the **scrubbed** value
    /// and `report.paths` on the durable `Event.redactions` (so the chain covers
    /// only the redacted form — the secret never enters the hash, the blob CAS,
    /// or the vector store), and (3) appends *this* `security.redaction` event so
    /// the *fact and location* of redaction are independently auditable. The
    /// event payload carries the JSON-pointer paths and per-detector tallies, but
    /// **never the secret** — only where and why a span was removed.
    ///
    /// Returns `None` when nothing was redacted (no event to emit), so the host
    /// can call this unconditionally.
    pub fn build_redaction_event(&self, session_id: SessionId) -> Option<NewEvent> {
        if self.is_clean() {
            return None;
        }
        let detectors: Vec<_> = self
            .redactions
            .iter()
            .map(|r| {
                serde_json::json!({
                    "detector": r.pattern_name,
                    "occurrences": r.occurrences,
                })
            })
            .collect();
        let total: usize = self.redactions.iter().map(|r| r.occurrences).sum();
        Some(NewEvent::system(
            session_id,
            "security.redaction",
            serde_json::json!({
                // RFC 6901 JSON-pointer paths of the scrubbed leaves — mirrors
                // what the host writes into Event.redactions.
                "paths": self.paths,
                // Per-detector tallies (why each span was scrubbed). No secret.
                "detectors": detectors,
                "total_spans": total,
            }),
        ))
    }
}

/// A known-format secret detector (compiled regex + a name).
#[derive(Clone)]
pub struct PatternDetector {
    pub name: String,
    pub regex: Regex,
}

impl std::fmt::Debug for PatternDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatternDetector")
            .field("name", &self.name)
            .finish()
    }
}

/// The redaction engine. Holds the ordered pattern detectors plus an entropy
/// threshold for the generic catch-all.
#[derive(Debug, Clone)]
pub struct Redactor {
    detectors: Vec<PatternDetector>,
    /// Tokens of at least this length whose Shannon entropy (bits/char) meets
    /// [`Self::entropy_threshold`] are redacted by the generic detector.
    entropy_min_len: usize,
    entropy_threshold: f64,
    /// Length+entropy dial for the **single-class** catch-all (item 3): a token
    /// of one character class (e.g. all-lowercase base64) is only redacted if it
    /// is at least this long AND at least [`Self::single_class_entropy`]
    /// bits/char. Set well above the prose floor so an all-lowercase secret can't
    /// hide behind the two-class `looks_secretish` gate, while ordinary long
    /// lowercase words (rare, and low-entropy) stay untouched.
    single_class_min_len: usize,
    single_class_entropy: f64,
    entropy_enabled: bool,
}

impl Default for Redactor {
    fn default() -> Self {
        Self {
            detectors: builtin_detectors().to_vec(),
            // Tuned so realistic 32+ char base64/hex secrets trip it while
            // ordinary prose words / short identifiers do not. English text
            // sits well under ~4.0 bits/char per token; a random 40-char
            // base64 blob is ~5.5–6.0.
            entropy_min_len: 24,
            entropy_threshold: 4.0,
            // Single-class dial. A long all-lowercase random base64/base36 blob
            // draws from ~26+ symbols → ~4.5–4.7 bits/char, while a hex sha draws
            // from only 16 symbols (≤4.0 bits/char ceiling) and a decimal id from
            // 10 (≤3.32). Setting the floor at 4.2 cleanly separates a lowercase
            // secret from a commit hash / numeric id, and sits above the prose
            // ceiling so ordinary lowercase words never trip it.
            single_class_min_len: 32,
            single_class_entropy: 4.2,
            entropy_enabled: true,
        }
    }
}

impl Redactor {
    /// Pattern detectors only (entropy off) — for callers that want zero
    /// false-positives on high-entropy-but-benign data (hashes, UUIDs).
    pub fn patterns_only() -> Self {
        Self {
            entropy_enabled: false,
            ..Self::default()
        }
    }

    pub fn with_entropy(mut self, min_len: usize, threshold: f64) -> Self {
        self.entropy_min_len = min_len;
        self.entropy_threshold = threshold;
        self.entropy_enabled = true;
        self
    }

    /// Tune the **single-class** catch-all dial (item 3): the minimum length and
    /// bits/char a single-character-class token (e.g. all-lowercase base64) must
    /// reach to be redacted even though it fails the two-class `looks_secretish`
    /// gate. Higher values = fewer false positives, more risk a single-class
    /// secret slips through.
    pub fn with_single_class(mut self, min_len: usize, entropy: f64) -> Self {
        self.single_class_min_len = min_len;
        self.single_class_entropy = entropy;
        self
    }

    /// The generic-detector decision for one token: redact if either
    ///   * it is mixed-class (≥2 of upper/lower/digit) and clears the standard
    ///     `entropy_min_len` / `entropy_threshold` gate, OR
    ///   * it is **single-class** but long enough and high-entropy enough to be a
    ///     credential rather than a word (item 3 — catches all-lowercase base64).
    fn is_high_entropy_secret(&self, token: &str) -> bool {
        let h = shannon_entropy(token);
        if looks_secretish(token)
            && token.len() >= self.entropy_min_len
            && h >= self.entropy_threshold
        {
            return true;
        }
        // Single-class branch: one character class only, but conspicuously long
        // and high-entropy. A pure-hex sha or all-decimal id has too few distinct
        // symbols to clear `single_class_entropy`, so commit hashes survive.
        token.len() >= self.single_class_min_len && h >= self.single_class_entropy
    }

    /// Register an extra detector (the `secret-detector` policy-plugin seam,
    /// §7 — tighten-only: plugins can add detections, never remove).
    pub fn with_detector(mut self, name: impl Into<String>, regex: Regex) -> Self {
        self.detectors.push(PatternDetector {
            name: name.into(),
            regex,
        });
        self
    }

    /// Scrub a flat string. Pattern detectors run first (most specific), then
    /// the entropy catch-all over whatever survives.
    pub fn redact(&self, input: &str) -> RedactionReport {
        let mut text = input.to_string();
        let mut tally: Vec<Redaction> = Vec::new();

        for det in &self.detectors {
            let mut count = 0usize;
            text = det
                .regex
                .replace_all(&text, |_: &regex::Captures| {
                    count += 1;
                    marker(&det.name)
                })
                .into_owned();
            if count > 0 {
                tally.push(Redaction {
                    pattern_name: det.name.clone(),
                    replacement: marker(&det.name),
                    occurrences: count,
                });
            }
        }

        if self.entropy_enabled {
            let (scrubbed, count) = self.redact_entropy(&text);
            if count > 0 {
                text = scrubbed;
                tally.push(Redaction {
                    pattern_name: "entropy".to_string(),
                    replacement: marker("entropy"),
                    occurrences: count,
                });
            }
        }

        RedactionReport {
            text,
            redactions: tally,
        }
    }

    /// Generic high-entropy token detector. Splits on whitespace and common
    /// delimiters, keeping the secret-like core; replaces any token that is long
    /// enough and high-entropy enough, and isn't already a redaction marker.
    fn redact_entropy(&self, input: &str) -> (String, usize) {
        let mut count = 0usize;
        // Walk the string, copying through, replacing qualifying runs of
        // "secret-ish" characters (alnum + a few base64/url-safe symbols).
        let mut out = String::with_capacity(input.len());
        let mut token = String::new();
        // `is_high_entropy_secret` owns ALL length/entropy gating (both the
        // mixed-class and single-class branches carry their own length floor), so
        // the closure only guards against re-redacting an existing marker.
        let flush = |token: &mut String, out: &mut String, count: &mut usize, this: &Redactor| {
            if !token.is_empty() {
                if !token.starts_with(MARKER_OPEN) && this.is_high_entropy_secret(token) {
                    out.push_str(&marker("entropy"));
                    *count += 1;
                } else {
                    out.push_str(token);
                }
                token.clear();
            }
        };
        for ch in input.chars() {
            if is_token_char(ch) {
                token.push(ch);
            } else {
                flush(&mut token, &mut out, &mut count, self);
                out.push(ch);
            }
        }
        flush(&mut token, &mut out, &mut count, self);
        (out, count)
    }

    /// Scrub every string leaf of a JSON value, returning the scrubbed value and
    /// the JSON-pointer paths of redacted leaves (§4.8). Object keys are NOT
    /// scrubbed (a key is structural, not content); only values.
    pub fn redact_json(&self, value: &Value) -> JsonRedactionReport {
        let mut paths = Vec::new();
        let mut tally: Vec<Redaction> = Vec::new();
        let scrubbed = self.scrub_value(value, String::new(), &mut paths, &mut tally);
        JsonRedactionReport {
            value: scrubbed,
            paths,
            redactions: tally,
        }
    }

    fn scrub_value(
        &self,
        value: &Value,
        pointer: String,
        paths: &mut Vec<String>,
        tally: &mut Vec<Redaction>,
    ) -> Value {
        match value {
            Value::String(s) => {
                let report = self.redact(s);
                if !report.is_clean() {
                    paths.push(if pointer.is_empty() {
                        "".to_string()
                    } else {
                        pointer
                    });
                    merge_tally(tally, report.redactions);
                    Value::String(report.text)
                } else {
                    Value::String(s.clone())
                }
            }
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.scrub_value(v, format!("{pointer}/{i}"), paths, tally))
                    .collect(),
            ),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| {
                        let child = format!("{pointer}/{}", escape_pointer_token(k));
                        (k.clone(), self.scrub_value(v, child, paths, tally))
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

fn merge_tally(tally: &mut Vec<Redaction>, more: Vec<Redaction>) {
    for r in more {
        if let Some(existing) = tally.iter_mut().find(|e| e.pattern_name == r.pattern_name) {
            existing.occurrences += r.occurrences;
        } else {
            tally.push(r);
        }
    }
}

/// RFC 6901: `~` → `~0`, `/` → `~1`.
fn escape_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

fn is_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '_' | '-' | '=' | '.')
}

/// A token that is *all* digits or *all* lowercase hex of a "nice" length is
/// likely an id/hash, not a credential; require some mixed-class content so we
/// don't redact every git SHA or large integer. (Pattern detectors still catch
/// real secrets that happen to be hex/base64.)
fn looks_secretish(token: &str) -> bool {
    let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
    let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    // Need at least two of {upper, lower, digit} — rules out pure-hex shas,
    // decimal ids, and uppercase-only constants.
    [has_upper, has_lower, has_digit]
        .iter()
        .filter(|b| **b)
        .count()
        >= 2
}

/// Shannon entropy in bits per character.
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    let mut n = 0usize;
    for b in s.bytes() {
        counts[b as usize] += 1;
        n += 1;
    }
    let n = n as f64;
    let mut h = 0.0;
    for &c in counts.iter() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h
}

/// The built-in pattern detectors (§4.8). Ordered most-specific first.
fn builtin_detectors() -> &'static [PatternDetector] {
    static DETECTORS: OnceLock<Vec<PatternDetector>> = OnceLock::new();
    DETECTORS.get_or_init(|| {
        let mut v = Vec::new();
        let mut add = |name: &str, pat: &str| {
            v.push(PatternDetector {
                name: name.to_string(),
                regex: Regex::new(pat).expect("builtin redaction pattern compiles"),
            });
        };
        // PEM private-key block (whole block, multi-line).
        add(
            "pem_private_key",
            r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        );
        // AWS access key id (AKIA/ASIA + 16 base32 chars).
        add("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b");
        // AWS secret access key, when introduced by an obvious key= context.
        add(
            "aws_secret_key",
            r#"(?i)aws_secret_access_key["']?\s*[:=]\s*["']?[A-Za-z0-9/+=]{40}"#,
        );
        // GitHub PATs (classic + fine-grained + app/refresh tokens).
        add("github_pat", r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b");
        add(
            "github_fine_grained_pat",
            r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b",
        );
        // GitLab PAT.
        add("gitlab_pat", r"\bglpat-[A-Za-z0-9_\-]{20,}\b");
        // Slack token.
        add("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b");
        // JWT: three base64url segments separated by dots; header starts eyJ.
        add(
            "jwt",
            r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b",
        );
        // Generic "sk-"/"rk-" style provider keys (kept from the old toy, widened).
        add("provider_key", r"\b[sr]k-[A-Za-z0-9]{20,}\b");
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_aws_access_key() {
        let r = Redactor::default().redact("export AWS_KEY=AKIAIOSFODNN7EXAMPLE done");
        assert!(
            r.text.contains("\u{00AB}redacted:aws_access_key\u{00BB}"),
            "{}",
            r.text
        );
        assert!(r
            .redactions
            .iter()
            .any(|x| x.pattern_name == "aws_access_key"));
        assert!(!r.text.contains("AKIA"));
    }

    #[test]
    fn redacts_github_pat() {
        let token = format!("ghp_{}", "a".repeat(36));
        let r = Redactor::default().redact(&format!("token={token}"));
        assert!(
            r.text.contains("\u{00AB}redacted:github_pat\u{00BB}"),
            "{}",
            r.text
        );
    }

    #[test]
    fn redacts_jwt() {
        let jwt = "eyJhbGciOiJIUzI1Ni1.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4f";
        let r = Redactor::default().redact(&format!("auth {jwt} end"));
        assert!(
            r.text.contains("\u{00AB}redacted:jwt\u{00BB}"),
            "{}",
            r.text
        );
    }

    #[test]
    fn redacts_pem_block() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj...\nabcDEF123==\n-----END RSA PRIVATE KEY-----";
        let r = Redactor::default().redact(&format!("key:\n{pem}\nrest"));
        assert!(
            r.text.contains("\u{00AB}redacted:pem_private_key\u{00BB}"),
            "{}",
            r.text
        );
        assert!(!r.text.contains("PRIVATE KEY-----\nMIIB"));
    }

    #[test]
    fn entropy_catches_unknown_high_entropy_token() {
        // No known prefix, but high-entropy mixed-class blob → entropy detector.
        let secret = "Zk9Qm2Xp7Lv3Rt8Wf1Yc6Nb4Hd0Sg5Aj"; // 33 chars, mixed
        let r = Redactor::default().redact(&format!("password is {secret} ok"));
        assert!(
            r.text.contains("\u{00AB}redacted:entropy\u{00BB}"),
            "got: {}",
            r.text
        );
    }

    #[test]
    fn marker_uses_guillemets_and_lowercase_detector(/* item 1 */) {
        // The marker must be «redacted:<detector>» with U+00AB/U+00BB guillemets
        // and a LOWERCASE detector name (bible §4.8), never the old ASCII
        // <<redacted:...>> form.
        let m = marker("AWS_Access_Key");
        assert_eq!(m, "\u{00AB}redacted:aws_access_key\u{00BB}");
        assert!(m.starts_with(MARKER_OPEN) && m.ends_with(MARKER_CLOSE));
        assert!(!m.contains("<<") && !m.contains(">>"));
        // A plugin-registered detector with mixed case is lowercased in output.
        let r = Redactor::patterns_only()
            .with_detector("MyCorp_Token", Regex::new(r"\bMYC-[0-9]{6}\b").unwrap())
            .redact("see MYC-123456 here");
        assert!(
            r.text.contains("\u{00AB}redacted:mycorp_token\u{00BB}"),
            "{}",
            r.text
        );
    }

    #[test]
    fn entropy_leaves_prose_and_ids_alone() {
        let prose = "the quick brown fox jumps over the lazy dog repeatedly today";
        let r = Redactor::default().redact(prose);
        assert!(r.is_clean(), "prose redacted: {:?}", r.redactions);

        // A long decimal id and a pure-hex sha should survive the entropy pass
        // (they fail looks_secretish), so we don't redact every commit hash.
        let idish = "0123456789012345678901234567 deadbeefdeadbeefdeadbeefdeadbeef";
        let r2 = Redactor::patterns_only().redact(idish);
        assert!(r2.is_clean());
    }

    #[test]
    fn redact_json_emits_pointer_paths() {
        let r = Redactor::default();
        let payload = serde_json::json!({
            "output": {
                "stdout": format!("ghp_{}", "b".repeat(36)),
                "exit": 0
            },
            "args": ["clean", "AKIAIOSFODNN7EXAMPLE"]
        });
        let report = r.redact_json(&payload);
        assert!(!report.is_clean());
        assert!(
            report.paths.contains(&"/output/stdout".to_string()),
            "{:?}",
            report.paths
        );
        assert!(
            report.paths.contains(&"/args/1".to_string()),
            "{:?}",
            report.paths
        );
        // Clean leaves untouched; non-string leaves preserved.
        assert_eq!(report.value["output"]["exit"], 0);
        assert_eq!(report.value["args"][0], "clean");
        assert!(report.value["output"]["stdout"]
            .as_str()
            .unwrap()
            .contains("\u{00AB}redacted:github_pat\u{00BB}"));
    }

    #[test]
    fn json_pointer_escapes_special_keys() {
        let r = Redactor::patterns_only();
        let payload = serde_json::json!({ "a/b": "AKIAIOSFODNN7EXAMPLE" });
        let report = r.redact_json(&payload);
        // `/` in the key becomes `~1` per RFC 6901.
        assert_eq!(report.paths, vec!["/a~1b".to_string()]);
    }

    #[test]
    fn multiple_occurrences_tallied() {
        let two = "AKIAIOSFODNN7EXAMPL1 and AKIAIOSFODNN7EXAMPL2";
        let r = Redactor::default().redact(two);
        let aws = r
            .redactions
            .iter()
            .find(|x| x.pattern_name == "aws_access_key")
            .unwrap();
        assert_eq!(aws.occurrences, 2);
    }

    #[test]
    fn entropy_catches_single_class_all_lowercase_secret(/* item 3 */) {
        // An all-lowercase, high-entropy base36-ish blob has only ONE character
        // class, so the two-class `looks_secretish` gate misses it — the
        // single-class branch must still redact it.
        // 35 chars, ALL lowercase letters (one character class), drawing from
        // ~24 distinct symbols → ~4.6 bits/char, over the 4.2 single-class floor.
        let secret = "qjxmfwbnzkdpvhsugtrclyaeoiqwrtmkxbv";
        assert!(
            !looks_secretish(secret),
            "test premise: token must be single-class"
        );
        assert!(
            shannon_entropy(secret) >= 4.2,
            "entropy {} too low for fixture",
            shannon_entropy(secret)
        );
        let r = Redactor::default().redact(&format!("api_key={secret}"));
        assert!(
            r.text.contains("\u{00AB}redacted:entropy\u{00BB}"),
            "single-class secret slipped: {}",
            r.text
        );

        // The single-class branch must NOT swallow a hex sha (16 symbols → ≤4.0
        // bits/char) or a long decimal id (10 symbols → ≤3.32).
        let sha = "a1b9c3d7e5f1a2b4c6d8e0f2a4b6c8d0e2f4a6b8"; // 40 hex chars
        let id = "0123456789012345678901234567890123456789"; // 40 digits
        let clean = Redactor::default().redact(&format!("{sha} {id}"));
        assert!(
            clean.is_clean(),
            "hex/decimal id redacted: {:?}",
            clean.redactions
        );
    }

    #[test]
    fn single_class_dial_is_tunable(/* item 3 dial */) {
        // Lowering the dial redacts a shorter single-class token; raising it past
        // the token's reach leaves it alone.
        let tok = "qjxmfwbnzkdpvhsugtrcl"; // 21 lowercase chars
        assert!(!looks_secretish(tok));
        let loosened = Redactor::default()
            .with_single_class(16, 3.5)
            .redact(&format!("x={tok}"));
        assert!(
            loosened.text.contains("\u{00AB}redacted:entropy\u{00BB}"),
            "{}",
            loosened.text
        );
        // Default dial (min_len 32) leaves the 21-char token untouched.
        let tight = Redactor::default().redact(&format!("x={tok}"));
        assert!(tight.is_clean(), "{:?}", tight.redactions);
    }

    #[test]
    fn build_redaction_event_shape(/* item 2 */) {
        use hide_core::ids::SessionId;
        let r = Redactor::default();
        let payload = serde_json::json!({
            "output": { "stdout": format!("ghp_{}", "b".repeat(36)) },
            "args": ["clean", "AKIAIOSFODNN7EXAMPLE"]
        });
        let report = r.redact_json(&payload);
        assert!(!report.is_clean());

        let session = SessionId::new();
        let ev = report
            .build_redaction_event(session)
            .expect("redacted payload yields an event");
        assert_eq!(ev.kind, "security.redaction");
        // Paths mirror what the host writes into Event.redactions.
        let paths = ev.payload["paths"].as_array().unwrap();
        assert!(paths.iter().any(|p| p == "/output/stdout"));
        assert!(paths.iter().any(|p| p == "/args/1"));
        // Per-detector tallies present; total span count present.
        assert!(ev.payload["detectors"].is_array());
        assert!(ev.payload["total_spans"].as_u64().unwrap() >= 2);
        // The event NEVER carries the secret itself.
        let serialized = serde_json::to_string(&ev.payload).unwrap();
        assert!(!serialized.contains("ghp_"));
        assert!(!serialized.contains("AKIA"));
    }

    #[test]
    fn build_redaction_event_none_when_clean(/* item 2 */) {
        use hide_core::ids::SessionId;
        let report = Redactor::default().redact_json(&serde_json::json!({ "ok": "hello world" }));
        assert!(report.is_clean());
        assert!(report.build_redaction_event(SessionId::new()).is_none());
    }
}
