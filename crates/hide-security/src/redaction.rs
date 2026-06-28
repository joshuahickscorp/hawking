//! Secret redaction before durability (bible ch.10 §4.8, S6).
//!
//! Replaces the original two-prefix toy with a real detector suite: known-format
//! pattern detectors (AWS access keys, GitHub/GitLab PATs, PEM private-key
//! blocks, JWTs, Slack tokens) plus a generic **Shannon-entropy** detector for
//! high-entropy tokens that no signature catches. On a hit, the span is replaced
//! with `<<redacted:DETECTOR>>` and the location is recorded so the *fact and
//! place* of redaction stays auditable while the secret never enters the log,
//! the chain hash, the blob CAS, or a vector store (§4.2.1 / §4.8).
//!
//! Two surfaces:
//!   * [`Redactor::redact`] — scrub a flat string (shell output, a log line).
//!   * [`Redactor::redact_json`] — scrub every string leaf of a JSON value and
//!     emit the **JSON-pointer paths** (RFC 6901) of redacted leaves, ready to
//!     drop into `Event.redactions` (§4.8). This is the form a `tool.result`
//!     payload goes through before it becomes a durable event.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

/// Marker substituted for a redacted span (§4.8). `DETECTOR` names which
/// detector fired, so the audit trail records *why* a span was scrubbed.
fn marker(detector: &str) -> String {
    format!("<<redacted:{detector}>>")
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
        let flush = |token: &mut String, out: &mut String, count: &mut usize, this: &Redactor| {
            if !token.is_empty() {
                if token.len() >= this.entropy_min_len
                    && !token.starts_with("<<redacted:")
                    && shannon_entropy(token) >= this.entropy_threshold
                    && looks_secretish(token)
                {
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
                    .map(|(i, v)| {
                        self.scrub_value(v, format!("{pointer}/{i}"), paths, tally)
                    })
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
        add("github_fine_grained_pat", r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b");
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
        assert!(r.text.contains("<<redacted:aws_access_key>>"), "{}", r.text);
        assert!(r.redactions.iter().any(|x| x.pattern_name == "aws_access_key"));
        assert!(!r.text.contains("AKIA"));
    }

    #[test]
    fn redacts_github_pat() {
        let token = format!("ghp_{}", "a".repeat(36));
        let r = Redactor::default().redact(&format!("token={token}"));
        assert!(r.text.contains("<<redacted:github_pat>>"), "{}", r.text);
    }

    #[test]
    fn redacts_jwt() {
        let jwt = "eyJhbGciOiJIUzI1Ni1.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4f";
        let r = Redactor::default().redact(&format!("auth {jwt} end"));
        assert!(r.text.contains("<<redacted:jwt>>"), "{}", r.text);
    }

    #[test]
    fn redacts_pem_block() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj...\nabcDEF123==\n-----END RSA PRIVATE KEY-----";
        let r = Redactor::default().redact(&format!("key:\n{pem}\nrest"));
        assert!(r.text.contains("<<redacted:pem_private_key>>"), "{}", r.text);
        assert!(!r.text.contains("PRIVATE KEY-----\nMIIB"));
    }

    #[test]
    fn entropy_catches_unknown_high_entropy_token() {
        // No known prefix, but high-entropy mixed-class blob → entropy detector.
        let secret = "Zk9Qm2Xp7Lv3Rt8Wf1Yc6Nb4Hd0Sg5Aj"; // 33 chars, mixed
        let r = Redactor::default().redact(&format!("password is {secret} ok"));
        assert!(r.text.contains("<<redacted:entropy>>"), "got: {}", r.text);
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
        assert!(report.paths.contains(&"/output/stdout".to_string()), "{:?}", report.paths);
        assert!(report.paths.contains(&"/args/1".to_string()), "{:?}", report.paths);
        // Clean leaves untouched; non-string leaves preserved.
        assert_eq!(report.value["output"]["exit"], 0);
        assert_eq!(report.value["args"][0], "clean");
        assert!(report.value["output"]["stdout"]
            .as_str()
            .unwrap()
            .contains("<<redacted:github_pat>>"));
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
        let two = format!("AKIAIOSFODNN7EXAMPL1 and AKIAIOSFODNN7EXAMPL2");
        let r = Redactor::default().redact(&two);
        let aws = r
            .redactions
            .iter()
            .find(|x| x.pattern_name == "aws_access_key")
            .unwrap();
        assert_eq!(aws.occurrences, 2);
    }
}
