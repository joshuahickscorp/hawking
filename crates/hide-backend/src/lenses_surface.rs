//! The three HIDE surfaces (lenses over one session).

use serde::{Deserialize, Serialize};

use crate::lenses::capability::SurfacePermissionSet;

/// YOU / CHAT / IDE — three lenses, one session. They differ in default
/// context and default permissions, not in intelligence or truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Private general-purpose multimodal personal AI.
    You,
    /// Repository-aware coding-agent workspace.
    Chat,
    /// Visual code and software-development environment.
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

    pub fn all() -> [Surface; 3] {
        [Self::You, Self::Chat, Self::Ide]
    }
}

impl std::fmt::Display for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default permission profile for a surface, matching
/// `evidence/hide/HIDE_YOU_SURFACE_AUTHORITY.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceDefaults {
    pub surface: Surface,
    /// Connector access description (metadata; closed set lives in permissions).
    pub connectors_policy: String,
    pub shell_policy: String,
    pub repo_write_policy: String,
    pub network_policy: String,
    /// Closed tool/connector set the surface may hold by default.
    pub permissions: SurfacePermissionSet,
}

impl SurfaceDefaults {
    /// YOU: connectors read-only, shell denied, repo write denied.
    pub fn you_default() -> Self {
        Self {
            surface: Surface::You,
            connectors_policy: "read-only".into(),
            shell_policy: "denied".into(),
            repo_write_policy: "denied".into(),
            network_policy: "explicit per session type".into(),
            // YOU may hold personal connectors (mail, calendar, vault) — read.
            permissions: SurfacePermissionSet::new(
                [
                    "connector.read",
                    "memory.read",
                    "research.read",
                    "object.read",
                ],
                ["gmail", "calendar", "personal_vault", "rss"],
            ),
        }
    }

    /// CHAT: repo-scoped connector read, shell under policy, repo write via effects.
    pub fn chat_default() -> Self {
        Self {
            surface: Surface::Chat,
            connectors_policy: "repo-scoped read".into(),
            shell_policy: "under policy".into(),
            repo_write_policy: "via effects".into(),
            network_policy: "denied by default".into(),
            // CHAT deliberately does NOT include personal connectors.
            permissions: SurfacePermissionSet::new(
                [
                    "repo.read",
                    "repo.write_effect",
                    "shell.under_policy",
                    "object.read",
                ],
                ["repo_index"],
            ),
        }
    }

    /// IDE: repo-scoped read, shell under policy, repo write via effects + visible diff.
    pub fn ide_default() -> Self {
        Self {
            surface: Surface::Ide,
            connectors_policy: "repo-scoped read".into(),
            shell_policy: "under policy".into(),
            repo_write_policy: "via effects with visible diff".into(),
            network_policy: "denied by default".into(),
            permissions: SurfacePermissionSet::new(
                [
                    "repo.read",
                    "repo.write_effect",
                    "shell.under_policy",
                    "diff.present",
                    "object.read",
                ],
                ["repo_index", "source_control"],
            ),
        }
    }

    pub fn for_surface(surface: Surface) -> Self {
        match surface {
            Surface::You => Self::you_default(),
            Surface::Chat => Self::chat_default(),
            Surface::Ide => Self::ide_default(),
        }
    }
}
