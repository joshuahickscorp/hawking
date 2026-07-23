//! Identity binding: the contract a capsule must satisfy to load into a runtime.
//!
//! A capsule is only meaningful under the exact conditions it was captured
//! under. Loading recurrent or cache bytes produced under one tokenizer, one
//! prompt ABI, or one security domain into a runtime configured differently is
//! not degraded, it is undefined. So a capsule carries an [`IdentityBinding`]
//! and refuses to bind unless every field matches the live runtime's binding.
//!
//! The comparison is strict equality on every field. A real runtime may later
//! choose to relax individual fields to compatibility ranges (for example a
//! prompt-ABI minimum rather than an exact match); that policy is
//! DEFERRED_MODEL_REQUIRED and lives with the runtime, not with these schemas.

use serde::{Deserialize, Serialize};

use crate::error::IncompatibleReason;

/// The set of identifiers a capsule must share with a live runtime to bind.
///
/// All fields are opaque identifiers compared by equality. The crate does not
/// interpret their contents; it only checks that the sealed side and the live
/// side agree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityBinding {
    pub model_weights_id: String,
    pub arch_id: String,
    pub tokenizer_id: String,
    pub prompt_abi_version: String,
    pub tool_registry_id: String,
    pub engine_build_id: String,
    pub security_domain: String,
}

impl IdentityBinding {
    /// Check whether a capsule carrying this binding can load into a runtime
    /// whose live binding is `live`.
    ///
    /// Returns `Ok(())` when every field agrees. Otherwise returns the typed
    /// reason for the first field that disagrees. The order is fixed and puts
    /// the security domain first, so a security-domain mismatch is reported in
    /// preference to any other difference.
    pub fn is_loadable(&self, live: &IdentityBinding) -> Result<(), IncompatibleReason> {
        if self.security_domain != live.security_domain {
            return Err(IncompatibleReason::SecurityDomain {
                capsule: self.security_domain.clone(),
                live: live.security_domain.clone(),
            });
        }
        if self.model_weights_id != live.model_weights_id {
            return Err(IncompatibleReason::ModelWeights {
                capsule: self.model_weights_id.clone(),
                live: live.model_weights_id.clone(),
            });
        }
        if self.arch_id != live.arch_id {
            return Err(IncompatibleReason::Arch {
                capsule: self.arch_id.clone(),
                live: live.arch_id.clone(),
            });
        }
        if self.tokenizer_id != live.tokenizer_id {
            return Err(IncompatibleReason::Tokenizer {
                capsule: self.tokenizer_id.clone(),
                live: live.tokenizer_id.clone(),
            });
        }
        if self.prompt_abi_version != live.prompt_abi_version {
            return Err(IncompatibleReason::PromptAbi {
                capsule: self.prompt_abi_version.clone(),
                live: live.prompt_abi_version.clone(),
            });
        }
        if self.tool_registry_id != live.tool_registry_id {
            return Err(IncompatibleReason::ToolRegistry {
                capsule: self.tool_registry_id.clone(),
                live: live.tool_registry_id.clone(),
            });
        }
        if self.engine_build_id != live.engine_build_id {
            return Err(IncompatibleReason::EngineBuild {
                capsule: self.engine_build_id.clone(),
                live: live.engine_build_id.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> IdentityBinding {
        IdentityBinding {
            model_weights_id: "weights-a".into(),
            arch_id: "arch-a".into(),
            tokenizer_id: "tok-a".into(),
            prompt_abi_version: "abi-1".into(),
            tool_registry_id: "reg-a".into(),
            engine_build_id: "build-a".into(),
            security_domain: "domain-a".into(),
        }
    }

    #[test]
    fn identical_bindings_are_loadable() {
        let a = sample();
        assert_eq!(a.is_loadable(&sample()), Ok(()));
    }

    #[test]
    fn each_field_mismatch_has_its_own_reason() {
        let base = sample();

        let mut live = sample();
        live.model_weights_id = "weights-b".into();
        assert_eq!(
            base.is_loadable(&live),
            Err(IncompatibleReason::ModelWeights {
                capsule: "weights-a".into(),
                live: "weights-b".into(),
            })
        );

        let mut live = sample();
        live.tokenizer_id = "tok-b".into();
        assert_eq!(
            base.is_loadable(&live),
            Err(IncompatibleReason::Tokenizer {
                capsule: "tok-a".into(),
                live: "tok-b".into(),
            })
        );

        let mut live = sample();
        live.security_domain = "domain-b".into();
        assert_eq!(
            base.is_loadable(&live),
            Err(IncompatibleReason::SecurityDomain {
                capsule: "domain-a".into(),
                live: "domain-b".into(),
            })
        );

        let mut live = sample();
        live.arch_id = "arch-b".into();
        assert!(matches!(
            base.is_loadable(&live),
            Err(IncompatibleReason::Arch { .. })
        ));

        let mut live = sample();
        live.prompt_abi_version = "abi-2".into();
        assert!(matches!(
            base.is_loadable(&live),
            Err(IncompatibleReason::PromptAbi { .. })
        ));

        let mut live = sample();
        live.tool_registry_id = "reg-b".into();
        assert!(matches!(
            base.is_loadable(&live),
            Err(IncompatibleReason::ToolRegistry { .. })
        ));

        let mut live = sample();
        live.engine_build_id = "build-b".into();
        assert!(matches!(
            base.is_loadable(&live),
            Err(IncompatibleReason::EngineBuild { .. })
        ));
    }

    #[test]
    fn security_domain_is_reported_before_other_mismatches() {
        let base = sample();
        let mut live = sample();
        live.security_domain = "domain-b".into();
        live.tokenizer_id = "tok-b".into();
        // Both differ; the security domain wins.
        assert!(matches!(
            base.is_loadable(&live),
            Err(IncompatibleReason::SecurityDomain { .. })
        ));
    }
}
