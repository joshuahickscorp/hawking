//! Prefix-cache discipline lint (W-F4-7).
//!
//! The `.tq`/Qwen prefix cache keys on a hash of the *static* prompt prefix
//! (system prompt, pinned repo spans). If dynamic content is spliced in BEFORE a
//! static segment, the prefix hash shifts every turn and the cache silently
//! misses -- a measured throughput cliff (~-36.7%). This pure lint asserts every
//! static segment precedes every dynamic one, so a request that would break the
//! cache is caught at construction instead of degrading silently.

/// One ordered piece of a composed prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSegment {
    pub label: String,
    /// True for cacheable/stable content (system prompt, pinned spans).
    pub is_static: bool,
}

impl PromptSegment {
    pub fn stat(label: impl Into<String>) -> Self {
        Self { label: label.into(), is_static: true }
    }
    pub fn dynamic(label: impl Into<String>) -> Self {
        Self { label: label.into(), is_static: false }
    }
}

/// Returns `Err(index)` of the first static segment that appears after a dynamic
/// one (a discipline violation), or `Ok(())` when the prefix is stable.
pub fn check_prefix_discipline(segments: &[PromptSegment]) -> Result<(), usize> {
    let mut seen_dynamic = false;
    for (i, seg) in segments.iter().enumerate() {
        if seg.is_static && seen_dynamic {
            return Err(i);
        }
        seen_dynamic |= !seg.is_static;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disciplined_prefix_passes() {
        let segs = vec![
            PromptSegment::stat("system"),
            PromptSegment::stat("repo"),
            PromptSegment::dynamic("turn"),
        ];
        assert_eq!(check_prefix_discipline(&segs), Ok(()));
    }

    #[test]
    fn static_after_dynamic_trips_the_lint() {
        let segs = vec![
            PromptSegment::stat("system"),
            PromptSegment::dynamic("turn"),
            PromptSegment::stat("pinned"), // violation at index 2
        ];
        assert_eq!(check_prefix_discipline(&segs), Err(2));
    }

    #[test]
    fn empty_and_all_dynamic_are_fine() {
        assert_eq!(check_prefix_discipline(&[]), Ok(()));
        assert_eq!(check_prefix_discipline(&[PromptSegment::dynamic("a")]), Ok(()));
    }
}
