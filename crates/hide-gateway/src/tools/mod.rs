//! Tool gateway (bible §16).
//!
//! Retrieves mutually-useful tool **sets**, enforces credentials/policy/session/
//! effects/health/version/schemas, and classifies tool defects separately from
//! model failures.

mod bundle;
mod classify;
mod enforce;
mod gateway;

pub use bundle::{kernel_bundle, BundleMember, KernelBundle, ToolBundle, ToolRef};
pub use classify::{
    classify_outcome, FailureClass, ModelFailureKind, OutcomeObservation, ToolDefectKind,
};
pub use enforce::{
    EffectBoundary, SessionAffinity, ToolEnforcement, ToolHealth, ToolHealthStatus, ToolPolicy,
    ToolVersion,
};
pub use gateway::{GrantedBundle, ToolGateway, ToolGatewayError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::classify::OutcomeObservation;
    use serde_json::json;

    fn demo_catalog() -> Vec<ToolRef> {
        vec![
            ToolRef {
                id: "fs.read".into(),
                name: "fs.read".into(),
                version: ToolVersion::parse("1.0.0"),
                effects: vec![EffectBoundary::Read],
                input_schema: json!({"type":"object","required":["path"]}),
                output_schema: Some(json!({"type":"object"})),
                requires_credential: None,
            },
            ToolRef {
                id: "profiler.sample".into(),
                name: "profiler.sample".into(),
                version: ToolVersion::parse("2.1.0"),
                effects: vec![EffectBoundary::Execute],
                input_schema: json!({"type":"object"}),
                output_schema: None,
                requires_credential: None,
            },
            ToolRef {
                id: "compiler.invoke".into(),
                name: "compiler.invoke".into(),
                version: ToolVersion::parse("1.4.0"),
                effects: vec![EffectBoundary::Execute, EffectBoundary::Write],
                input_schema: json!({"type":"object","required":["target"]}),
                output_schema: None,
                requires_credential: None,
            },
            ToolRef {
                id: "bench.run".into(),
                name: "bench.run".into(),
                version: ToolVersion::parse("0.9.0"),
                effects: vec![EffectBoundary::Execute],
                input_schema: json!({"type":"object"}),
                output_schema: None,
                requires_credential: None,
            },
            ToolRef {
                id: "receipt.verify".into(),
                name: "receipt.verify".into(),
                version: ToolVersion::parse("1.0.0"),
                effects: vec![EffectBoundary::Read],
                input_schema: json!({"type":"object","required":["path"]}),
                output_schema: None,
                requires_credential: None,
            },
            ToolRef {
                id: "artifact.inspect".into(),
                name: "artifact.inspect".into(),
                version: ToolVersion::parse("1.0.0"),
                effects: vec![EffectBoundary::Read],
                input_schema: json!({"type":"object"}),
                output_schema: None,
                requires_credential: None,
            },
            ToolRef {
                id: "net.fetch".into(),
                name: "net.fetch".into(),
                version: ToolVersion::parse("1.0.0"),
                effects: vec![EffectBoundary::Network],
                input_schema: json!({"type":"object"}),
                output_schema: None,
                requires_credential: Some("hf_token".into()),
            },
        ]
    }

    #[test]
    fn kernel_bundle_is_a_set_not_isolated_tools() {
        let b = kernel_bundle();
        let names: Vec<_> = b.members.iter().map(|m| m.role.as_str()).collect();
        assert!(names.contains(&"source_reader"));
        assert!(names.contains(&"profiler"));
        assert!(names.contains(&"compiler"));
        assert!(names.contains(&"benchmark_runner"));
        assert!(names.contains(&"receipt_verifier"));
        assert!(names.contains(&"artifact_inspector"));
        assert!(b.members.len() >= 6);
        assert!(b.mutual_affinity > 0.5);
    }

    #[test]
    fn gateway_retrieves_bundle_and_enforces_policy() {
        let mut gw = ToolGateway::new();
        for t in demo_catalog() {
            gw.register_tool(t);
        }
        gw.register_bundle(kernel_bundle());
        gw.set_health(
            "fs.read",
            ToolHealth {
                status: ToolHealthStatus::Healthy,
                last_error: None,
            },
        );
        for id in [
            "profiler.sample",
            "compiler.invoke",
            "bench.run",
            "receipt.verify",
            "artifact.inspect",
        ] {
            gw.set_health(
                id,
                ToolHealth {
                    status: ToolHealthStatus::Healthy,
                    last_error: None,
                },
            );
        }

        let session = SessionAffinity::new("sess-1");
        let policy = ToolPolicy {
            max_effect: EffectBoundary::Execute,
            allow_network: false,
            require_healthy: true,
            profile: "maximum".into(),
        };

        let granted = gw
            .retrieve_bundle("kernel", &session, &policy)
            .expect("kernel bundle");
        assert_eq!(granted.bundle.id, "kernel");
        assert_eq!(granted.tools.len(), 6);
        assert!(granted
            .tools
            .iter()
            .all(|t| t.effects.iter().all(|e| e.rank() <= EffectBoundary::Execute.rank())));
    }

    #[test]
    fn network_tool_blocked_when_policy_disallows() {
        let mut gw = ToolGateway::new();
        for t in demo_catalog() {
            gw.register_tool(t);
        }
        gw.register_bundle(ToolBundle {
            id: "net".into(),
            name: "network probe".into(),
            members: vec![BundleMember {
                tool_id: "net.fetch".into(),
                role: "fetcher".into(),
                required: true,
            }],
            mutual_affinity: 1.0,
        });
        gw.set_health(
            "net.fetch",
            ToolHealth {
                status: ToolHealthStatus::Healthy,
                last_error: None,
            },
        );
        gw.grant_credential("sess-1", "hf_token");

        let session = SessionAffinity::new("sess-1");
        let policy = ToolPolicy {
            max_effect: EffectBoundary::Network,
            allow_network: false,
            require_healthy: true,
            profile: "sandbox".into(),
        };
        let err = gw.retrieve_bundle("net", &session, &policy).unwrap_err();
        assert!(
            matches!(err, ToolGatewayError::PolicyDenied { .. }),
            "expected policy deny, got {err:?}"
        );
    }

    #[test]
    fn unhealthy_required_member_fails_bundle_retrieval() {
        let mut gw = ToolGateway::new();
        for t in demo_catalog() {
            gw.register_tool(t);
        }
        gw.register_bundle(kernel_bundle());
        for id in [
            "fs.read",
            "profiler.sample",
            "compiler.invoke",
            "bench.run",
            "receipt.verify",
            "artifact.inspect",
        ] {
            let status = if id == "compiler.invoke" {
                ToolHealthStatus::Unhealthy
            } else {
                ToolHealthStatus::Healthy
            };
            gw.set_health(
                id,
                ToolHealth {
                    status,
                    last_error: if id == "compiler.invoke" {
                        Some("segfault".into())
                    } else {
                        None
                    },
                },
            );
        }

        let err = gw
            .retrieve_bundle(
                "kernel",
                &SessionAffinity::new("s"),
                &ToolPolicy {
                    max_effect: EffectBoundary::Execute,
                    allow_network: false,
                    require_healthy: true,
                    profile: "maximum".into(),
                },
            )
            .unwrap_err();
        assert!(matches!(err, ToolGatewayError::UnhealthyTool { .. }));
    }

    #[test]
    fn missing_credential_is_tool_side_not_model_side() {
        let mut gw = ToolGateway::new();
        for t in demo_catalog() {
            gw.register_tool(t);
        }
        gw.register_bundle(ToolBundle {
            id: "net".into(),
            name: "network".into(),
            members: vec![BundleMember {
                tool_id: "net.fetch".into(),
                role: "fetcher".into(),
                required: true,
            }],
            mutual_affinity: 1.0,
        });
        gw.set_health(
            "net.fetch",
            ToolHealth {
                status: ToolHealthStatus::Healthy,
                last_error: None,
            },
        );

        let err = gw
            .retrieve_bundle(
                "net",
                &SessionAffinity::new("sess-no-cred"),
                &ToolPolicy {
                    max_effect: EffectBoundary::Network,
                    allow_network: true,
                    require_healthy: true,
                    profile: "gate".into(),
                },
            )
            .unwrap_err();
        assert!(matches!(err, ToolGatewayError::MissingCredential { .. }));
    }

    #[test]
    fn classify_schema_violation_as_tool_defect() {
        let tool = ToolRef {
            id: "fs.read".into(),
            name: "fs.read".into(),
            version: ToolVersion::parse("1.0.0"),
            effects: vec![EffectBoundary::Read],
            input_schema: json!({"type":"object","required":["path"]}),
            output_schema: None,
            requires_credential: None,
        };
        let obs = OutcomeObservation {
            tool_ok: false,
            tool_error: Some("schema: missing required property path".into()),
            exit_code: None,
            model_expected_tool: Some("fs.read".into()),
            model_emitted_args_valid: false,
            timed_out: false,
            effect_violated: false,
        };
        let class = classify_outcome(&tool, &obs);
        assert!(
            matches!(class, FailureClass::ToolDefect { kind: ToolDefectKind::SchemaViolation, .. }),
            "{class:?}"
        );
    }

    #[test]
    fn classify_wrong_tool_choice_as_model_failure() {
        let tool = ToolRef {
            id: "fs.read".into(),
            name: "fs.read".into(),
            version: ToolVersion::parse("1.0.0"),
            effects: vec![EffectBoundary::Read],
            input_schema: json!({}),
            output_schema: None,
            requires_credential: None,
        };
        let obs = OutcomeObservation {
            tool_ok: true,
            tool_error: None,
            exit_code: Some(0),
            model_expected_tool: Some("fs.write".into()),
            model_emitted_args_valid: true,
            timed_out: false,
            effect_violated: false,
        };
        let class = classify_outcome(&tool, &obs);
        assert!(
            matches!(
                class,
                FailureClass::ModelFailure {
                    kind: ModelFailureKind::WrongTool,
                    ..
                }
            ),
            "{class:?}"
        );
    }

    #[test]
    fn classify_nonzero_exit_as_data_not_tool_defect_when_contract_says_so() {
        // Mirrors hide-core ToolResult: EXEC_NONZERO is data, not tool failure.
        let tool = ToolRef {
            id: "shell.run".into(),
            name: "shell.run".into(),
            version: ToolVersion::parse("1.0.0"),
            effects: vec![EffectBoundary::Execute],
            input_schema: json!({}),
            output_schema: None,
            requires_credential: None,
        };
        let obs = OutcomeObservation {
            tool_ok: true, // tool layer succeeded
            tool_error: None,
            exit_code: Some(1),
            model_expected_tool: Some("shell.run".into()),
            model_emitted_args_valid: true,
            timed_out: false,
            effect_violated: false,
        };
        let class = classify_outcome(&tool, &obs);
        assert!(
            matches!(class, FailureClass::SuccessWithData { .. }),
            "nonzero exit with ok:true is data: {class:?}"
        );
    }

    #[test]
    fn classify_timeout_as_tool_defect() {
        let tool = ToolRef {
            id: "bench.run".into(),
            name: "bench.run".into(),
            version: ToolVersion::parse("0.9.0"),
            effects: vec![EffectBoundary::Execute],
            input_schema: json!({}),
            output_schema: None,
            requires_credential: None,
        };
        let obs = OutcomeObservation {
            tool_ok: false,
            tool_error: Some("timeout".into()),
            exit_code: None,
            model_expected_tool: Some("bench.run".into()),
            model_emitted_args_valid: true,
            timed_out: true,
            effect_violated: false,
        };
        let class = classify_outcome(&tool, &obs);
        assert!(matches!(
            class,
            FailureClass::ToolDefect {
                kind: ToolDefectKind::Timeout,
                ..
            }
        ));
    }

    #[test]
    fn classify_effect_boundary_breach_as_tool_defect() {
        let tool = ToolRef {
            id: "fs.read".into(),
            name: "fs.read".into(),
            version: ToolVersion::parse("1.0.0"),
            effects: vec![EffectBoundary::Read],
            input_schema: json!({}),
            output_schema: None,
            requires_credential: None,
        };
        let obs = OutcomeObservation {
            tool_ok: false,
            tool_error: Some("effect Write not declared".into()),
            exit_code: None,
            model_expected_tool: Some("fs.read".into()),
            model_emitted_args_valid: true,
            timed_out: false,
            effect_violated: true,
        };
        let class = classify_outcome(&tool, &obs);
        assert!(matches!(
            class,
            FailureClass::ToolDefect {
                kind: ToolDefectKind::EffectBoundaryBreach,
                ..
            }
        ));
    }

    #[test]
    fn session_affinity_pins_bundle_to_session() {
        let mut gw = ToolGateway::new();
        for t in demo_catalog() {
            gw.register_tool(t);
        }
        gw.register_bundle(kernel_bundle());
        for id in [
            "fs.read",
            "profiler.sample",
            "compiler.invoke",
            "bench.run",
            "receipt.verify",
            "artifact.inspect",
        ] {
            gw.set_health(
                id,
                ToolHealth {
                    status: ToolHealthStatus::Healthy,
                    last_error: None,
                },
            );
        }
        let s1 = SessionAffinity::new("alpha");
        let policy = ToolPolicy {
            max_effect: EffectBoundary::Execute,
            allow_network: false,
            require_healthy: true,
            profile: "maximum".into(),
        };
        let g1 = gw.retrieve_bundle("kernel", &s1, &policy).unwrap();
        assert_eq!(g1.session_id, "alpha");
        // Re-retrieve binds same session; gateway records affinity.
        assert!(gw.session_has_bundle("alpha", "kernel"));
        assert!(!gw.session_has_bundle("beta", "kernel"));
    }
}
