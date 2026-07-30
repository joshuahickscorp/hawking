//! Per-object permissions, enforced at read time.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};

/// Which HIDE surface may see the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Private general-purpose personal AI.
    You,
    /// Coding-agent workspace.
    Chat,
    /// Visual dev environment.
    Ide,
}

impl Surface {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::You => "you",
            Self::Chat => "chat",
            Self::Ide => "ide",
        }
    }
}

/// Who is asking to read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reader {
    pub principal: String,
    pub surface: Surface,
}

/// Access control attached to every object record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectPermissions {
    /// Owner principal (always may read/write metadata).
    pub owner: String,
    /// Additional principals allowed to read.
    #[serde(default)]
    pub readers: Vec<String>,
    /// Surfaces this object is visible on. Empty = none.
    #[serde(default)]
    pub surfaces: Vec<Surface>,
    /// Whether selected derivatives may be compiled into model context.
    #[serde(default = "default_true")]
    pub allow_model_derivatives: bool,
    /// Whether raw/export paths are allowed (still requires RawBytesCap).
    #[serde(default)]
    pub allow_export: bool,
}

fn default_true() -> bool {
    true
}

impl ObjectPermissions {
    pub fn owner_only(owner: impl Into<String>, surfaces: Vec<Surface>) -> Self {
        Self {
            owner: owner.into(),
            readers: Vec::new(),
            surfaces,
            allow_model_derivatives: true,
            allow_export: false,
        }
    }

    pub fn allows_principal(&self, principal: &str) -> bool {
        principal == self.owner || self.readers.iter().any(|r| r == principal)
    }

    pub fn allows_surface(&self, surface: Surface) -> bool {
        self.surfaces.contains(&surface)
    }

    /// Read-time gate: principal + surface must both pass.
    pub fn check_read(&self, reader: &Reader) -> Result<()> {
        if !self.allows_principal(&reader.principal) {
            return Err(ObjectError::PermissionDenied {
                reason: format!(
                    "principal '{}' is not owner or listed reader",
                    reader.principal
                ),
            });
        }
        if !self.allows_surface(reader.surface) {
            return Err(ObjectError::PermissionDenied {
                reason: format!(
                    "surface '{}' is not permitted on this object",
                    reader.surface.as_str()
                ),
            });
        }
        Ok(())
    }

    pub fn check_model_derivatives(&self, reader: &Reader) -> Result<()> {
        self.check_read(reader)?;
        if !self.allow_model_derivatives {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_model_derivatives is false".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_reads_on_allowed_surface() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "alice".into(),
            surface: Surface::You,
        };
        assert!(p.check_read(&r).is_ok());
    }
    #[test]
    fn stranger_denied() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "bob".into(),
            surface: Surface::You,
        };
        assert!(matches!(
            p.check_read(&r),
            Err(ObjectError::PermissionDenied { .. })
        ));
    }
    #[test]
    fn wrong_surface_denied() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "alice".into(),
            surface: Surface::Chat,
        };
        assert!(matches!(
            p.check_read(&r),
            Err(ObjectError::PermissionDenied { .. })
        ));
    }
}
