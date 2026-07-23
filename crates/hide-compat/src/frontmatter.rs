//! A deliberately small YAML-subset reader for markdown frontmatter.
//!
//! Agent and skill definitions carry a `---` delimited frontmatter block with a
//! constrained shape: scalar keys, booleans, and lists (either block form with
//! `- item` lines or inline `[a, b, c]`). We do not want a full YAML dependency
//! for a model-free compatibility layer, so this parser handles exactly that
//! subset and nothing more. It is deterministic and total: malformed input never
//! panics, it simply yields whatever keys it could recognise.

use std::collections::BTreeMap;

/// A recognised frontmatter value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Scalar(String),
    Bool(bool),
    List(Vec<String>),
    Null,
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Scalar(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// A list view. A bare scalar is treated as a one-element list so callers
    /// that accept either `tools: Read` or `tools: [Read, Write]` behave the same.
    pub fn as_list(&self) -> Vec<String> {
        match self {
            Value::List(v) => v.clone(),
            Value::Scalar(s) => vec![s.clone()],
            _ => Vec::new(),
        }
    }
}

/// A parsed frontmatter block. Keys are ordered for stable iteration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    pub map: BTreeMap<String, Value>,
}

impl Frontmatter {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    pub fn str(&self, key: &str) -> Option<String> {
        self.map.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.map.get(key).and_then(|v| v.as_bool())
    }

    pub fn list(&self, key: &str) -> Vec<String> {
        self.map.get(key).map(|v| v.as_list()).unwrap_or_default()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
}

/// Split a markdown document into its optional frontmatter block and the body
/// that follows it. If there is no leading `---` fence the whole document is
/// returned as the body and the frontmatter is `None`.
pub fn split(content: &str) -> (Option<Frontmatter>, String) {
    // A frontmatter block must be the very first line of the file. Allow a BOM
    // and trailing whitespace on the fence line, nothing else before it.
    let stripped = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut lines = stripped.lines();
    let first = match lines.next() {
        Some(l) => l,
        None => return (None, String::new()),
    };
    if first.trim() != "---" {
        return (None, content.to_string());
    }

    let mut block = String::new();
    let mut closed = false;
    let mut body = String::new();
    let mut in_body = false;
    for line in lines {
        if in_body {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if line.trim() == "---" {
            closed = true;
            in_body = true;
            continue;
        }
        block.push_str(line);
        block.push('\n');
    }

    if !closed {
        // Unterminated fence: treat the whole thing as body, no frontmatter.
        return (None, content.to_string());
    }

    (Some(parse_block(&block)), body)
}

/// Parse just a frontmatter block (already stripped of its `---` fences).
pub fn parse_block(text: &str) -> Frontmatter {
    let mut map: BTreeMap<String, Value> = BTreeMap::new();
    let mut cur_key: Option<String> = None;
    let mut cur_list: Vec<String> = Vec::new();

    // Flush a pending block-list key into the map.
    fn flush(map: &mut BTreeMap<String, Value>, key: Option<String>, list: &mut Vec<String>) {
        if let Some(k) = key {
            if list.is_empty() {
                map.insert(k, Value::Null);
            } else {
                map.insert(k, Value::List(std::mem::take(list)));
            }
        }
    }

    for raw in text.lines() {
        let line = strip_inline_comment(raw);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let indented = raw
            .chars()
            .next()
            .map(|c| c == ' ' || c == '\t')
            .unwrap_or(false);

        if indented && trimmed.starts_with('-') {
            // Block-list item belonging to the current key.
            if cur_key.is_some() {
                let item = trimmed[1..].trim();
                if !item.is_empty() {
                    cur_list.push(unquote(item));
                }
                continue;
            }
            // A dash with no owning key: ignore.
            continue;
        }

        // A new key starts here; flush any pending block list first.
        flush(&mut map, cur_key.take(), &mut cur_list);

        let colon = match trimmed.find(':') {
            Some(i) => i,
            None => continue,
        };
        let key = trimmed[..colon].trim().to_string();
        if key.is_empty() {
            continue;
        }
        let rest = trimmed[colon + 1..].trim();

        if rest.is_empty() {
            // Possibly a block list; wait for indented `- ` lines.
            cur_key = Some(key);
        } else if rest.starts_with('[') && rest.ends_with(']') {
            let inner = &rest[1..rest.len() - 1];
            let items: Vec<String> = inner
                .split(',')
                .map(|s| unquote(s.trim()))
                .filter(|s| !s.is_empty())
                .collect();
            map.insert(key, Value::List(items));
        } else {
            map.insert(key, scalar_value(rest));
        }
    }

    flush(&mut map, cur_key.take(), &mut cur_list);
    Frontmatter { map }
}

fn scalar_value(rest: &str) -> Value {
    match rest {
        "true" | "True" | "yes" => Value::Bool(true),
        "false" | "False" | "no" => Value::Bool(false),
        "null" | "~" => Value::Null,
        _ => Value::Scalar(unquote(rest)),
    }
}

/// Strip a trailing `# comment` that is not inside a quoted string. Matching
/// YAML, a `#` is a comment only when it follows whitespace (or starts the
/// line); a `#` glued to non-whitespace (e.g. `a#b`) is kept. A hex value must
/// therefore be quoted (`color: "#ffffff"`) to survive, exactly as in YAML.
fn strip_inline_comment(line: &str) -> String {
    let bytes: Vec<char> = line.chars().collect();
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_ws = true; // start-of-line counts as preceding whitespace
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double && prev_ws => {
                return bytes[..i].iter().collect::<String>();
            }
            _ => {}
        }
        prev_ws = c == ' ' || c == '\t';
    }
    line.to_string()
}

/// Remove a single pair of matching surrounding quotes if present.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        let first = bytes[0];
        let last = bytes[s.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_frontmatter_and_body() {
        let doc = "---\nname: foo\n---\nbody line\nmore\n";
        let (fm, body) = split(doc);
        let fm = fm.expect("frontmatter present");
        assert_eq!(fm.str("name").as_deref(), Some("foo"));
        assert_eq!(body, "body line\nmore\n");
    }

    #[test]
    fn no_frontmatter_returns_whole_body() {
        let doc = "just a body\nno fence\n";
        let (fm, body) = split(doc);
        assert!(fm.is_none());
        assert_eq!(body, doc);
    }

    #[test]
    fn parses_inline_and_block_lists() {
        let block = "tools: [Read, Write]\nskills:\n  - a\n  - b\n";
        let fm = parse_block(block);
        assert_eq!(fm.list("tools"), vec!["Read", "Write"]);
        assert_eq!(fm.list("skills"), vec!["a", "b"]);
    }

    #[test]
    fn parses_booleans_and_quotes() {
        let block = "user-invocable: true\ndisable-model-invocation: false\ndescription: \"hello world\"\n";
        let fm = parse_block(block);
        assert_eq!(fm.bool("user-invocable"), Some(true));
        assert_eq!(fm.bool("disable-model-invocation"), Some(false));
        assert_eq!(fm.str("description").as_deref(), Some("hello world"));
    }

    #[test]
    fn empty_key_becomes_null_not_list() {
        let block = "memory:\nname: x\n";
        let fm = parse_block(block);
        assert_eq!(fm.get("memory"), Some(&Value::Null));
        assert_eq!(fm.str("name").as_deref(), Some("x"));
    }

    #[test]
    fn strips_inline_comment_and_quoted_hash_survives() {
        let block = "quoted: \"#ffffff\"\nname: foo # trailing\nglued: a#b\nbare: #ffffff\n";
        let fm = parse_block(block);
        // Quoted hash survives.
        assert_eq!(fm.str("quoted").as_deref(), Some("#ffffff"));
        // Trailing comment after whitespace is stripped.
        assert_eq!(fm.str("name").as_deref(), Some("foo"));
        // A hash glued to non-whitespace is not a comment.
        assert_eq!(fm.str("glued").as_deref(), Some("a#b"));
        // A bare hash after whitespace is a YAML comment: value becomes null.
        assert_eq!(fm.get("bare"), Some(&Value::Null));
    }
}
