//! The deterministic data model that flows through a program.
//!
//! [`Value`] is a JSON-shaped value with a fixed, deterministic serialization:
//! objects are backed by a [`BTreeMap`] so keys are always emitted in sorted
//! order, which is what makes byte-identical output across runs possible. There
//! is deliberately no handle to the outside world in this type: it is pure data.
//!
//! Provenance rides along inside records. A record (a [`Value::Map`]) may carry
//! a reserved [`CITATIONS_KEY`] field holding a list of [`Citation`]s. The
//! citation-preserving operators read and merge that field so that evidence is
//! never silently dropped as data is transformed.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

/// The reserved record field that carries provenance. A [`Value::Map`] with
/// this key holds a [`Value::List`] of citation objects.
pub const CITATIONS_KEY: &str = "@citations";

/// A piece of evidence that a record was derived from. Structured so it can be
/// re-verified later against a content-addressed store (the `digest`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// The source the evidence came from: a handle name, a path, a uri, a
    /// commit, etc. Opaque to the runtime.
    pub source: String,
    /// Where inside the source: a line range, a symbol, a byte span. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// A content digest of the cited bytes, for later re-verification. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    /// A short human-readable excerpt. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

impl Citation {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            locator: None,
            digest: None,
            snippet: None,
        }
    }

    pub fn with_locator(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }

    pub fn with_digest(mut self, digest: impl Into<String>) -> Self {
        self.digest = Some(digest.into());
        self
    }

    /// A stable key used to dedup citations during merge. Two citations with the
    /// same key are treated as the same piece of evidence.
    pub fn dedup_key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.source,
            self.locator.as_deref().unwrap_or(""),
            self.digest.as_deref().unwrap_or(""),
        )
    }

    fn to_value(&self) -> Value {
        let mut m = BTreeMap::new();
        m.insert("source".to_string(), Value::Str(self.source.clone()));
        if let Some(l) = &self.locator {
            m.insert("locator".to_string(), Value::Str(l.clone()));
        }
        if let Some(d) = &self.digest {
            m.insert("digest".to_string(), Value::Str(d.clone()));
        }
        if let Some(s) = &self.snippet {
            m.insert("snippet".to_string(), Value::Str(s.clone()));
        }
        Value::Map(m)
    }

    /// Parse one citation from a `{source, locator?, digest?, snippet?}` map.
    /// Returns `None` if the value is not a map or lacks a `source` string.
    pub fn from_value(v: &Value) -> Option<Citation> {
        let m = v.as_map()?;
        let source = m.get("source").and_then(Value::as_str)?.to_string();
        Some(Citation {
            source,
            locator: m.get("locator").and_then(Value::as_str).map(str::to_string),
            digest: m.get("digest").and_then(Value::as_str).map(str::to_string),
            snippet: m.get("snippet").and_then(Value::as_str).map(str::to_string),
        })
    }

    /// Parse a [`Value::List`] of citation maps, skipping any that do not parse.
    pub fn list_from(v: &Value) -> Vec<Citation> {
        match v.as_list() {
            Some(items) => items.iter().filter_map(Citation::from_value).collect(),
            None => Vec::new(),
        }
    }
}

/// A JSON-shaped value. Maps are sorted so serialization is canonical.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Truthiness for predicates: false / null / 0 / empty string / empty
    /// collection are falsey; everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }

    /// Follow a dotted path of keys into nested maps. Returns `None` if any
    /// segment is missing or a non-map is traversed.
    pub fn get_path(&self, path: &[String]) -> Option<&Value> {
        let mut cur = self;
        for key in path {
            cur = cur.as_map()?.get(key)?;
        }
        Some(cur)
    }

    /// A deterministic canonical string key for grouping / joining / dedup. It
    /// is the canonical JSON encoding, so it is type-sensitive (integer `1` and
    /// string `"1"` do not collide).
    pub fn canonical_key(&self) -> String {
        serde_json::to_string(self).expect("Value always serializes")
    }

    /// A cheap estimate of the memory footprint of this value, in bytes. Used by
    /// the memory limiter; it does not have to be exact, only monotone and
    /// deterministic.
    pub fn estimated_bytes(&self) -> u64 {
        match self {
            Value::Null | Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 8,
            Value::Str(s) => s.len() as u64,
            Value::List(v) => 8 + v.iter().map(Value::estimated_bytes).sum::<u64>(),
            Value::Map(m) => {
                8 + m
                    .iter()
                    .map(|(k, v)| k.len() as u64 + v.estimated_bytes())
                    .sum::<u64>()
            }
        }
    }

    fn type_rank(&self) -> u8 {
        match self {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 2,
            Value::Str(_) => 3,
            Value::List(_) => 4,
            Value::Map(_) => 5,
        }
    }

    /// A deterministic total order across all value shapes. Numbers compare
    /// numerically (integers and floats interleave); unlike types order by a
    /// fixed type rank. Used by `rank`.
    pub fn total_cmp(&self, other: &Value) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (a, b) if a.type_rank() == 2 && b.type_rank() == 2 => {
                let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
                x.partial_cmp(&y).unwrap_or(Ordering::Equal)
            }
            (Value::Str(a), Value::Str(b)) => a.cmp(b),
            (Value::List(a), Value::List(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    match x.total_cmp(y) {
                        Ordering::Equal => continue,
                        non_eq => return non_eq,
                    }
                }
                a.len().cmp(&b.len())
            }
            (Value::Map(a), Value::Map(b)) => {
                // Compare by canonical key encoding for a stable, total order.
                a.iter()
                    .map(|(k, v)| (k, v.canonical_key()))
                    .collect::<Vec<_>>()
                    .cmp(&b.iter().map(|(k, v)| (k, v.canonical_key())).collect::<Vec<_>>())
            }
            (a, b) => a.type_rank().cmp(&b.type_rank()),
        }
    }

    // --- citation helpers -------------------------------------------------

    /// Read the citations attached to a record. Returns an empty vector for a
    /// non-map or a record without the reserved field.
    pub fn citations(&self) -> Vec<Citation> {
        let Some(m) = self.as_map() else {
            return Vec::new();
        };
        let Some(Value::List(items)) = m.get(CITATIONS_KEY) else {
            return Vec::new();
        };
        items.iter().filter_map(Citation::from_value).collect()
    }

    /// Return a copy of this record with `extra` citations merged into its
    /// reserved field (deduplicated, stable order). A no-op on a non-map value.
    pub fn with_merged_citations(&self, extra: &[Citation]) -> Value {
        let Value::Map(m) = self else {
            return self.clone();
        };
        let mut merged = self.citations();
        let mut seen: std::collections::BTreeSet<String> =
            merged.iter().map(Citation::dedup_key).collect();
        for c in extra {
            if seen.insert(c.dedup_key()) {
                merged.push(c.clone());
            }
        }
        let mut out = m.clone();
        out.insert(
            CITATIONS_KEY.to_string(),
            Value::List(merged.iter().map(Citation::to_value).collect()),
        );
        Value::Map(out)
    }
}

// -- convenience conversions --------------------------------------------------

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v as i64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::List(v)
    }
}
impl From<BTreeMap<String, Value>> for Value {
    fn from(v: BTreeMap<String, Value>) -> Self {
        Value::Map(v)
    }
}

/// Build a [`Value::Map`] from an iterator of `(key, value)` pairs.
pub fn map_of<I, K, V>(pairs: I) -> Value
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<Value>,
{
    Value::Map(pairs.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
}

// -- serde: emit plain JSON via a serde_json::Value bridge --------------------

impl Value {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::List(v) => serde_json::Value::Array(v.iter().map(Value::to_json).collect()),
            Value::Map(m) => {
                // Insert in sorted (BTreeMap) order so the encoding is canonical
                // regardless of whether serde_json preserves insertion order.
                let mut obj = serde_json::Map::new();
                for (k, v) in m {
                    obj.insert(k.clone(), v.to_json());
                }
                serde_json::Value::Object(obj)
            }
        }
    }

    fn from_json(j: serde_json::Value) -> Value {
        match j {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(u) = n.as_u64() {
                    Value::Int(u as i64)
                } else {
                    Value::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Value::Str(s),
            serde_json::Value::Array(a) => {
                Value::List(a.into_iter().map(Value::from_json).collect())
            }
            serde_json::Value::Object(o) => {
                Value::Map(o.into_iter().map(|(k, v)| (k, Value::from_json(v))).collect())
            }
        }
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let j = serde_json::Value::deserialize(deserializer)?;
        Ok(Value::from_json(j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_json_with_sorted_keys() {
        let v = map_of([("b", Value::Int(2)), ("a", Value::Int(1))]);
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn roundtrips_through_json() {
        let v = map_of([
            ("n", Value::Null),
            ("f", Value::Float(1.5)),
            ("l", Value::List(vec![Value::Int(1), Value::Str("x".into())])),
        ]);
        let s = serde_json::to_string(&v).unwrap();
        let back: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn total_cmp_interleaves_int_and_float() {
        assert_eq!(Value::Int(1).total_cmp(&Value::Float(1.5)), Ordering::Less);
        assert_eq!(Value::Float(2.0).total_cmp(&Value::Int(2)), Ordering::Equal);
    }

    #[test]
    fn citations_roundtrip_and_merge_dedup() {
        let rec = map_of([("id", Value::Int(1))]);
        let c1 = Citation::new("file.read").with_locator("L1-L3");
        let merged = rec.with_merged_citations(&[c1.clone(), c1.clone()]);
        assert_eq!(merged.citations(), vec![c1.clone()]);
        // Merging an already-present citation is idempotent.
        let again = merged.with_merged_citations(&[c1.clone()]);
        assert_eq!(again.citations(), vec![c1]);
    }
}
