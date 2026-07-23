//! The re-review dependency model (Bible Book IX, sec 29).
//!
//! A verification receipt is only valid as long as the source it covers has not
//! changed. Given a set of prior receipts and the set of file paths a new change
//! touched, this module returns exactly the receipts whose scope INTERSECTS the
//! change: those are invalidated and must be re-run. Receipts whose scope is
//! disjoint from the change stay valid and are not re-run.

use crate::receipt::VerificationReceipt;

/// Normalize a path for comparison: strip a leading `./` and drop a trailing `/`
/// so directory and file spellings compare cleanly.
fn norm(p: &str) -> &str {
    let p = p.strip_prefix("./").unwrap_or(p);
    p.strip_suffix('/').unwrap_or(p)
}

/// True if two paths refer to the same file, or one is a directory that contains
/// the other. So a receipt scoping `crates/a/src` is invalidated by a change to
/// `crates/a/src/lib.rs`, and vice versa, but not by a change to `crates/ab`.
pub fn paths_intersect(a: &str, b: &str) -> bool {
    let a = norm(a);
    let b = norm(b);
    if a == b {
        return true;
    }
    let under = |child: &str, parent: &str| {
        child.starts_with(parent) && child.as_bytes().get(parent.len()) == Some(&b'/')
    };
    under(a, b) || under(b, a)
}

/// The receipts whose scope intersects ANY changed path (they must be re-run),
/// in the order they appear in `receipts`. Receipts with no intersecting scope
/// are omitted.
pub fn invalidated_receipts<'a>(
    receipts: &'a [VerificationReceipt],
    changed: &[String],
) -> Vec<&'a VerificationReceipt> {
    receipts
        .iter()
        .filter(|r| {
            r.scope
                .iter()
                .any(|s| changed.iter().any(|c| paths_intersect(s, c)))
        })
        .collect()
}

/// The `verification_id`s of the invalidated receipts. A convenience over
/// [`invalidated_receipts`].
pub fn invalidated_ids(receipts: &[VerificationReceipt], changed: &[String]) -> Vec<String> {
    invalidated_receipts(receipts, changed)
        .into_iter()
        .map(|r| r.verification_id.clone())
        .collect()
}
