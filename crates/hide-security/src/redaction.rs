use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Redaction {
    pub pattern_name: String,
    pub replacement: String,
    pub occurrences: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionReport {
    pub text: String,
    pub redactions: Vec<Redaction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Redactor {
    pub patterns: Vec<SecretPattern>,
}

impl Default for Redactor {
    fn default() -> Self {
        Self {
            patterns: vec![
                SecretPattern {
                    name: "openai_key_like".to_string(),
                    prefix: "sk-".to_string(),
                    min_len: 20,
                },
                SecretPattern {
                    name: "github_token_like".to_string(),
                    prefix: "ghp_".to_string(),
                    min_len: 20,
                },
            ],
        }
    }
}

impl Redactor {
    pub fn redact(&self, input: &str) -> RedactionReport {
        let mut text = input.to_string();
        let mut redactions = Vec::new();
        for pattern in &self.patterns {
            let mut occurrences = 0usize;
            let mut output = Vec::new();
            for token in text.split_whitespace() {
                if token.starts_with(&pattern.prefix) && token.len() >= pattern.min_len {
                    output.push("<REDACTED>");
                    occurrences += 1;
                } else {
                    output.push(token);
                }
            }
            if occurrences > 0 {
                text = output.join(" ");
                redactions.push(Redaction {
                    pattern_name: pattern.name.clone(),
                    replacement: "<REDACTED>".to_string(),
                    occurrences,
                });
            }
        }
        RedactionReport { text, redactions }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretPattern {
    pub name: String,
    pub prefix: String,
    pub min_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redactor_masks_key_like_tokens() {
        let report = Redactor::default().redact("token sk-abcdefghijklmnopqrstuvwxyz");
        assert!(report.text.contains("<REDACTED>"));
        assert_eq!(report.redactions.len(), 1);
    }
}
