//! Per-object retention, enforced at read time.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};

/// How long an object remains readable.
///
/// The store never silently drops objects on TTL expiry: reads fail visibly
/// with [`ObjectError::RetentionDenied`]. Explicit GC is a separate path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Readable only while the named session is considered live.
    Session { session_id: String },
    /// Survives sessions; explicit delete only.
    Durable,
    /// Wall-clock expiry (ms since epoch). After this, reads fail.
    Ttl { expires_at_ms: u64 },
    /// Never auto-expire; only owner delete.
    ExplicitDeleteOnly,
}

impl RetentionPolicy {
    pub fn durable() -> Self {
        Self::Durable
    }

    pub fn session(session_id: impl Into<String>) -> Self {
        Self::Session {
            session_id: session_id.into(),
        }
    }

    pub fn ttl_until(expires_at_ms: u64) -> Self {
        Self::Ttl { expires_at_ms }
    }

    /// Read-time check.
    ///
    /// - `now_ms`: wall clock for TTL
    /// - `live_session`: if `Some`, session-scoped objects for that id are live
    pub fn check_readable(&self, now_ms: u64, live_session: Option<&str>) -> Result<()> {
        match self {
            Self::Durable | Self::ExplicitDeleteOnly => Ok(()),
            Self::Session { session_id } => match live_session {
                Some(live) if live == session_id => Ok(()),
                Some(_) => Err(ObjectError::RetentionDenied {
                    reason: format!(
                        "session-scoped object for '{session_id}' not live (active={live_session:?})"
                    ),
                }),
                None => Err(ObjectError::RetentionDenied {
                    reason: format!(
                        "session-scoped object for '{session_id}' has no live session"
                    ),
                }),
            },
            Self::Ttl { expires_at_ms } => {
                if now_ms >= *expires_at_ms {
                    Err(ObjectError::RetentionDenied {
                        reason: format!(
                            "ttl expired at {expires_at_ms} (now={now_ms})"
                        ),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ttl_blocks_after_expiry() {
        let r = RetentionPolicy::ttl_until(1000);
        assert!(r.check_readable(999, None).is_ok());
        assert!(matches!(
            r.check_readable(1000, None),
            Err(ObjectError::RetentionDenied { .. })
        ));
    }
    #[test]
    fn session_requires_live() {
        let r = RetentionPolicy::session("ses_1");
        assert!(r.check_readable(0, Some("ses_1")).is_ok());
        assert!(r.check_readable(0, Some("ses_other")).is_err());
        assert!(r.check_readable(0, None).is_err());
    }
}
