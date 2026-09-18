//! Resident, deterministic tool-catalog ranking for Gravity.
//!
//! Python remains the authority for schemas, permissions, and dispatch.  This
//! module owns only the repeated pure ranking operation over a catalog that the
//! Python registry explicitly supplies.  That separation makes the hot path
//! native without creating a second permission system.

use crate::{Error, Result};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GravityToolEntry {
    pub name: String,
    pub search_text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GravityToolCatalog {
    entries: Vec<GravityToolEntry>,
}

impl GravityToolCatalog {
    /// Replace the whole catalog atomically at the protocol boundary.
    pub fn from_entries(mut entries: Vec<GravityToolEntry>) -> Result<Self> {
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        let mut names = BTreeSet::new();
        for entry in &entries {
            if entry.name.is_empty() || entry.name.chars().any(char::is_whitespace) {
                return Err(Error::Gravity(format!(
                    "invalid tool name {:?}",
                    entry.name
                )));
            }
            if entry.search_text.is_empty() {
                return Err(Error::Gravity(format!(
                    "tool {} has empty search text",
                    entry.name
                )));
            }
            if !names.insert(entry.name.clone()) {
                return Err(Error::Gravity(format!(
                    "duplicate tool name {}",
                    entry.name
                )));
            }
        }
        Ok(Self { entries })
    }

    /// Match Python's stable catalog scoring for pre-expanded query terms.
    ///
    /// Callers expand semantic aliases before crossing the protocol boundary;
    /// vocabulary policy therefore remains in the HCLI registry, while this
    /// function stays a minimal deterministic ranking primitive.
    pub fn query(&self, terms: &[String], max_results: usize) -> Result<(Vec<String>, usize)> {
        if max_results == 0 || max_results > 32 {
            return Err(Error::Gravity(
                "tool catalog max_results must be 1..=32".into(),
            ));
        }
        let folded = terms
            .iter()
            .filter_map(|term| {
                let term = term.trim().to_ascii_lowercase();
                (!term.is_empty()).then_some(term)
            })
            .collect::<Vec<_>>();
        let mut scored = self
            .entries
            .iter()
            .filter_map(|entry| {
                let name = entry.name.to_ascii_lowercase();
                let text = entry.search_text.to_ascii_lowercase();
                let score = folded.iter().fold(0u32, |score, term| {
                    score
                        + if *term == name {
                            100
                        } else if name.contains(term) {
                            40
                        } else if text.contains(term) {
                            10
                        } else {
                            0
                        }
                });
                (score > 0 || folded.is_empty()).then_some((score, entry.name.as_str()))
            })
            .collect::<Vec<_>>();
        if folded.is_empty() {
            scored.sort_by(|left, right| left.1.cmp(right.1));
        } else {
            scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1)));
        }
        let match_count = scored.len();
        Ok((
            scored
                .into_iter()
                .take(max_results)
                .map(|(_, name)| name.to_owned())
                .collect(),
            match_count,
        ))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> GravityToolCatalog {
        GravityToolCatalog::from_entries(vec![
            GravityToolEntry {
                name: "git.status".into(),
                search_text: "git.status inspect repository".into(),
            },
            GravityToolEntry {
                name: "lake.catalog".into(),
                search_text: "lake.catalog ModelLake exact specimen metadata".into(),
            },
        ])
        .unwrap()
    }

    #[test]
    fn exact_name_outranks_description_and_ties_are_lexical() {
        assert_eq!(
            catalog().query(&["catalog".into()], 12).unwrap(),
            (vec!["lake.catalog".to_owned()], 1)
        );
        assert_eq!(
            catalog().query(&[], 12).unwrap(),
            (vec!["git.status".to_owned(), "lake.catalog".to_owned()], 2)
        );
    }

    #[test]
    fn invalid_catalog_or_limit_fails_closed() {
        assert!(GravityToolCatalog::from_entries(vec![GravityToolEntry {
            name: "bad name".into(),
            search_text: "x".into()
        }])
        .is_err());
        assert!(catalog().query(&[], 0).is_err());
    }
}
