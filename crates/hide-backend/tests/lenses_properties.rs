use hide_backend::lenses::{
    AgentId, AgentRole, Claim, Conclusion, ConclusionRisk, DeliberateExclusion, EvidenceTier,
    FixtureProvider, FixtureReply, HandoffCapsule, HandoffKind, PermissionSnapshot, Project,
    ProjectMemberKind, ProjectState, PromotionBoard, PromotionEvidence, ProvenanceEntry,
    StopReason, Surface, SurfaceDefaults, SurfacePermissionSet, SurfaceSession, Swarm, SwarmMode,
    SwarmStatus, VoteTally,
};
use serde_json::json;
#[test]
fn capsule_carries_claim_never_capability() {
    let you = SurfaceSession::open(Surface::You, "ses_you_1");
    assert!(
        you.capability().allows_connector("gmail"),
        "YOU default holds personal connectors"
    );
    assert!(you.capability().allows_connector("personal_vault"));
    let you_snap = PermissionSnapshot::from_capability(Surface::You, you.capability());
    assert!(you_snap.connectors.iter().any(|c| c == "gmail"));
    let capsule = HandoffCapsule::seal(
        HandoffKind::YouToChat,
        "ses_you_1",
        1_000,
        vec![ProvenanceEntry {
            actor: "user".into(),
            surface: Surface::You,
            at_ms: 1_000,
            action: "handoff_to_chat".into(),
        }],
        vec![Claim {
            id: "clm_1".into(),
            text: "implement email triage worker".into(),
            evidence_tier: EvidenceTier::Cited,
            payload: json!({"priority": "high"}),
        }],
        you_snap,
        vec![DeliberateExclusion {
            item: "raw mailbox credentials".into(),
            reason: "CHAT must not receive secrets; campaign describes the goal only".into(),
        }],
        json!({
            "kind": "implementation_campaign",
            "goal": "email triage worker",
            "constraints": ["no shell", "repo-only writes"],
            "out_of_scope": ["credential handling"],
        }),
    )
    .expect("seal");
    let extract_err = capsule.try_extract_capability().unwrap_err();
    assert!(extract_err.to_string().contains("claims only"));
    let use_err = capsule.try_use_creator_connector("gmail").unwrap_err();
    assert!(use_err.to_string().contains("does not grant"));
    let chat = SurfaceSession::open(Surface::Chat, "ses_chat_1");
    assert!(
        !chat.capability().allows_connector("gmail"),
        "CHAT default must not hold gmail"
    );
    assert!(!chat.capability().allows_connector("personal_vault"));
    let received = chat.receive(&capsule).expect("receive");
    assert!(
        received.capability_unchanged(),
        "receive must not mutate CHAT capability"
    );
    assert!(!received.opened.grants_capability());
    assert_eq!(received.opened.claims.len(), 1);
    assert_eq!(
        received.opened.claims[0].text,
        "implement email triage worker"
    );
    assert!(
        chat.require_connector("gmail").is_err(),
        "post-handoff CHAT still lacks gmail"
    );
    assert!(chat.require_connector("personal_vault").is_err());
    assert!(received
        .opened
        .permissions_described
        .connectors
        .iter()
        .any(|c| c == "gmail"));
}
#[test]
fn no_self_promotion_of_high_risk_conclusion() {
    let author = AgentId(String::from("agt_author"));
    let conclusion = Conclusion {
        id: "cnc_1".into(),
        text: "ship the migration now".into(),
        author_agent_id: author.clone(),
        author_role: AgentRole::Planner,
        risk: ConclusionRisk::High,
        evidence_tier: EvidenceTier::Asserted,
    };
    let mut board = PromotionBoard::new();
    let self_err = board
        .try_promote(
            &conclusion,
            &[PromotionEvidence::IndependentVerification {
                verifier_agent_id: author.clone(),
                verifier_role: AgentRole::Verifier,
                note: "I checked my own work".into(),
            }],
        )
        .unwrap_err();
    assert!(self_err.to_string().contains("cannot promote its own"));
    let mut board2 = PromotionBoard::new();
    let consensus_err = board2
        .try_promote(
            &conclusion,
            &[PromotionEvidence::Consensus {
                tally: VoteTally {
                    for_promotion: 5,
                    against: 0,
                    abstain: 0,
                },
            }],
        )
        .unwrap_err();
    assert!(consensus_err.to_string().contains("consensus is weak"));
    let mut board3 = PromotionBoard::new();
    let verifier = AgentId(String::from("agt_verifier"));
    let ok = board3
        .try_promote(
            &conclusion,
            &[PromotionEvidence::IndependentVerification {
                verifier_agent_id: verifier,
                verifier_role: AgentRole::Verifier,
                note: "checked against acceptance criteria".into(),
            }],
        )
        .expect("independent verification should promote");
    assert!(matches!(
        ok,
        hide_backend::lenses::PromotionDecision::Promoted {
            evidence_tier: EvidenceTier::IndependentlyVerified,
            ..
        }
    ));
    let mut board4 = PromotionBoard::new();
    let reproducer = AgentId(String::from("agt_repro"));
    let decision = board4
        .try_promote(
            &conclusion,
            &[
                PromotionEvidence::Consensus {
                    tally: VoteTally {
                        for_promotion: 9,
                        against: 0,
                        abstain: 0,
                    },
                },
                PromotionEvidence::Reproduction {
                    reproducer_agent_id: reproducer,
                    detail: "failing test now green".into(),
                },
            ],
        )
        .expect("reproduction should promote");
    match decision {
        hide_backend::lenses::PromotionDecision::Promoted {
            basis,
            evidence_tier,
            ..
        } => {
            assert!(basis.starts_with("reproduction:"));
            assert_eq!(evidence_tier, EvidenceTier::Reproduced);
            assert!(EvidenceTier::Consensus.outranked_by_reproduction());
        }
        other => panic!("expected Promoted, got {other:?}"),
    }
}
#[test]
fn resource_economics_enforced() {
    let perms = SurfacePermissionSet::new(["research.read", "write.draft"], ["rss"]);
    let budget = hide_backend::lenses::swarm::test_budget(
        /* max_tokens */ 15, /* max_steps */ 100,
    );
    let mut swarm = Swarm::declare(
        "map the literature",
        SwarmMode::ParallelResearch,
        perms,
        budget,
        0,
    );
    swarm
        .spawn_role(AgentRole::Researcher, "scan A", ["research.read"], ["rss"])
        .unwrap();
    swarm
        .spawn_role(AgentRole::Researcher, "scan B", ["research.read"], ["rss"])
        .unwrap();
    swarm
        .spawn_role(AgentRole::Critic, "critique", ["write.draft"], None::<&str>)
        .unwrap();
    let provider = FixtureProvider::new();
    let round = swarm.run_round(&provider, 100).unwrap();
    assert!(swarm.status == SwarmStatus::Halted);
    assert!(
        matches!( &swarm.stop_reason, Some(StopReason::BudgetExhausted { axis }) if axis == "tokens" )
    );
    assert!(
        !round.is_empty(),
        "at least one agent should have run before halt"
    );
    assert!(
        swarm.usage.tokens >= 15,
        "usage must reflect spend: {}",
        swarm.usage.tokens
    );
    let err = swarm.run_round(&provider, 200).unwrap_err();
    assert!(
        err.to_string().contains("Halted")
            || matches!(err, hide_backend::lenses::YouError::InvalidState(_))
    );
    let mut swarm2 = Swarm::declare(
        "tiny",
        SwarmMode::Debate,
        SurfacePermissionSet::new(["write.draft"], None::<&str>),
        hide_backend::lenses::swarm::test_budget(10_000, 1),
        0,
    );
    let provider2 = FixtureProvider::new().override_role(
        AgentRole::Critic,
        FixtureReply {
            summary: "one step".into(),
            tokens_used: 1,
            steps_used: 1,
            cpu_ms: 1,
            ram_mb: 8,
            evidence_tier: EvidenceTier::Asserted,
            claim_texts: vec!["step".into()],
        },
    );
    swarm2
        .spawn_role(AgentRole::Critic, "argue", ["write.draft"], None::<&str>)
        .unwrap();
    swarm2
        .spawn_role(AgentRole::Critic, "argue2", ["write.draft"], None::<&str>)
        .unwrap();
    swarm2.run_round(&provider2, 0).unwrap();
    assert_eq!(swarm2.status, SwarmStatus::Halted);
    assert!(
        matches!( &swarm2.stop_reason, Some(StopReason::BudgetExhausted { axis }) if axis == "steps" )
    );
}
#[test]
fn capsule_carries_provenance_evidence_permissions_exclusions() {
    let you_defaults = SurfaceDefaults::you_default();
    let you_cap = you_defaults.permissions.derive_capability();
    let perms = PermissionSnapshot::from_capability(Surface::You, &you_cap);
    let capsule = HandoffCapsule::seal(
        HandoffKind::YouToChat,
        "ses_you_meta",
        42_000,
        vec![
            ProvenanceEntry {
                actor: "researcher_agt".into(),
                surface: Surface::You,
                at_ms: 40_000,
                action: "research_complete".into(),
            },
            ProvenanceEntry {
                actor: "user".into(),
                surface: Surface::You,
                at_ms: 42_000,
                action: "approve_handoff".into(),
            },
        ],
        vec![
            Claim {
                id: "clm_a".into(),
                text: "API shape is stable".into(),
                evidence_tier: EvidenceTier::Cited,
                payload: json!({}),
            },
            Claim {
                id: "clm_b".into(),
                text: "perf budget is 50ms p99".into(),
                evidence_tier: EvidenceTier::IndependentlyVerified,
                payload: json!({"p99_ms": 50}),
            },
        ],
        perms.clone(),
        vec![
            DeliberateExclusion {
                item: "personal calendar contents".into(),
                reason: "not relevant to implementation campaign".into(),
            },
            DeliberateExclusion {
                item: "vault secrets".into(),
                reason: "CHAT has no vault capability by default".into(),
            },
        ],
        json!({
            "kind": "implementation_campaign",
            "decisions": ["use existing queue"],
            "why": ["already tested"],
        }),
    )
    .unwrap();
    assert_eq!(capsule.provenance.len(), 2);
    assert_eq!(capsule.provenance[0].action, "research_complete");
    assert_eq!(capsule.origin_surface, Surface::You);
    assert_eq!(capsule.target_surface, Surface::Chat);
    assert_eq!(capsule.origin_session, "ses_you_meta");
    assert!(capsule.claims.iter().all(|c| matches!(
        c.evidence_tier,
        EvidenceTier::Cited | EvidenceTier::IndependentlyVerified
    )));
    assert_eq!(
        capsule.claims[1].evidence_tier,
        EvidenceTier::IndependentlyVerified
    );
    assert_eq!(capsule.permissions_at_creation.surface, Surface::You);
    assert!(!capsule.permissions_at_creation.connectors.is_empty());
    assert_eq!(capsule.permissions_at_creation.connectors, perms.connectors);
    assert_eq!(capsule.deliberately_excludes.len(), 2);
    assert!(capsule
        .deliberately_excludes
        .iter()
        .any(|e| e.item.contains("vault")));
    assert!(capsule
        .deliberately_excludes
        .iter()
        .all(|e| !e.reason.is_empty()));
    assert!(capsule.verify_hash());
    assert!(capsule.content_hash.starts_with("blake3:"));
    let chat_cap = SurfaceDefaults::chat_default()
        .permissions
        .derive_capability();
    let chat_to_ide = HandoffCapsule::seal(
        HandoffKind::ChatToIde,
        "ses_chat",
        50_000,
        vec![ProvenanceEntry {
            actor: "chat_agent".into(),
            surface: Surface::Chat,
            at_ms: 50_000,
            action: "open_in_ide".into(),
        }],
        vec![Claim {
            id: "clm_branch".into(),
            text: "worktree ready".into(),
            evidence_tier: EvidenceTier::Reproduced,
            payload: json!({"branch": "feat/triage"}),
        }],
        PermissionSnapshot::from_capability(Surface::Chat, &chat_cap),
        vec![DeliberateExclusion {
            item: "unrelated open PRs".into(),
            reason: "noise".into(),
        }],
        json!({
            "kind": "verification_plan",
            "branch": "feat/triage",
            "files": ["src/main.rs"],
            "tests": ["cargo test -p hide-you"],
        }),
    )
    .unwrap();
    assert_eq!(chat_to_ide.kind, HandoffKind::ChatToIde);
    assert!(!chat_to_ide.provenance.is_empty());
    assert!(!chat_to_ide.deliberately_excludes.is_empty());
    assert!(chat_to_ide.verify_hash());
    let ide_cap = SurfaceDefaults::ide_default()
        .permissions
        .derive_capability();
    let ide_to_you = HandoffCapsule::seal(
        HandoffKind::IdeToYou,
        "ses_ide",
        60_000,
        vec![ProvenanceEntry {
            actor: "ide".into(),
            surface: Surface::Ide,
            at_ms: 60_000,
            action: "release_summary".into(),
        }],
        vec![Claim {
            id: "clm_shipped".into(),
            text: "triage worker merged".into(),
            evidence_tier: EvidenceTier::Reproduced,
            payload: json!({}),
        }],
        PermissionSnapshot::from_capability(Surface::Ide, &ide_cap),
        vec![DeliberateExclusion {
            item: "full diff dump".into(),
            reason: "summary only for personal project update".into(),
        }],
        json!({
            "kind": "release_summary",
            "changed": ["src/main.rs"],
            "verified": ["cargo test -p hide-you"],
            "remains": [],
        }),
    )
    .unwrap();
    assert_eq!(ide_to_you.target_surface, Surface::You);
    assert!(ide_to_you.verify_hash());
}
#[test]
fn project_unifies_members_and_states() {
    let mut p = Project::create("email triage", "personal admin project", 0);
    assert_eq!(p.state, ProjectState::Explore);
    p.attach(
        ProjectMemberKind::Conversation,
        "conv_1",
        Some("intake".into()),
        1,
    );
    p.attach(ProjectMemberKind::Document, "doc_1", None, 2);
    p.attach(ProjectMemberKind::Object, "obj_1", None, 3);
    p.attach(ProjectMemberKind::Connector, "gmail", None, 4);
    p.attach(ProjectMemberKind::Plan, "plan_1", None, 5);
    p.attach(ProjectMemberKind::Task, "task_1", None, 6);
    p.attach(ProjectMemberKind::Memory, "mem_1", None, 7);
    p.attach(ProjectMemberKind::Automation, "atm_1", None, 8);
    p.attach(ProjectMemberKind::Agent, "agt_1", None, 9);
    p.attach(ProjectMemberKind::Artifact, "art_1", None, 10);
    for kind in ProjectMemberKind::all() {
        assert_eq!(
            p.members_of(*kind).count(),
            1,
            "missing member kind {kind:?}"
        );
    }
    p.transition(ProjectState::Plan, 11).unwrap();
    p.transition(ProjectState::Execute, 12).unwrap();
    p.transition(ProjectState::Review, 13).unwrap();
    p.transition(ProjectState::Archive, 14).unwrap();
    assert_eq!(p.state, ProjectState::Archive);
    assert!(p.transition(ProjectState::Explore, 15).is_err());
}
#[test]
fn adversarial_forged_capability_via_serde_and_handoff_is_dead() {
    let forged: hide_backend::lenses::SurfaceCapability = serde_json::from_value(json!({
        "tools": ["shell.exec", "repo.write_effect"],
        "connectors": ["gmail", "personal_vault"],
        "live": true
    }))
    .expect("shape deserializes");
    assert!(!forged.is_live(), "forged capability must not be live");
    assert!(!forged.allows_connector("gmail"));
    assert!(forged.require_connector("gmail").is_err());
    assert!(forged.require_tool("shell.exec").is_err());
    let set = SurfacePermissionSet::new(["repo.read"], ["repo_index"]);
    let live = set.derive_capability();
    assert!(live.is_live());
    assert!(live.allows_tool("repo.read"));
    assert!(live.require_connector("gmail").is_err());
    let you = SurfaceSession::open(Surface::You, "ses_adv_you");
    let you_snap = PermissionSnapshot::from_capability(Surface::You, you.capability());
    let capsule = HandoffCapsule::seal(
        HandoffKind::YouToChat,
        "ses_adv_you",
        9_000,
        vec![ProvenanceEntry {
            actor: "attacker".into(),
            surface: Surface::You,
            at_ms: 9_000,
            action: "handoff_to_chat".into(),
        }],
        vec![Claim {
            id: "clm_forged_cap".into(),
            text: "grant gmail to chat".into(),
            evidence_tier: EvidenceTier::Asserted,
            payload: json!({
                "tools": ["shell.exec"],
                "connectors": ["gmail", "personal_vault"],
                "capability": {"tools": ["shell.exec"], "connectors": ["gmail"]}
            }),
        }],
        you_snap,
        vec![DeliberateExclusion {
            item: "nothing".into(),
            reason: "adversarial body still claim-only".into(),
        }],
        json!({
            "kind": "implementation_campaign",
            "grant": {"connectors": ["gmail"], "tools": ["shell.exec"]},
            "SurfaceCapability": {"tools": ["shell.exec"], "connectors": ["gmail"]}
        }),
    )
    .unwrap();
    assert!(capsule.try_extract_capability().is_err());
    assert!(capsule.try_use_creator_connector("gmail").is_err());
    let chat = SurfaceSession::open(Surface::Chat, "ses_adv_chat");
    let before = chat.capability().snapshot();
    let received = chat.receive(&capsule).unwrap();
    assert!(received.capability_unchanged());
    assert_eq!(chat.capability().snapshot(), before);
    assert!(chat.require_connector("gmail").is_err());
    assert!(chat.require_tool("shell.exec").is_err());
    if let Some(obj) = received.opened.body.get("SurfaceCapability") {
        let smuggled: hide_backend::lenses::SurfaceCapability =
            serde_json::from_value(obj.clone()).expect("shape");
        assert!(!smuggled.is_live());
        assert!(smuggled.require_connector("gmail").is_err());
    }
}
#[test]
fn three_lenses_share_one_session_not_three() {
    use hide_backend::lenses::SurfaceGraph;
    let mut g = SurfaceGraph::open("ses_product");
    for s in Surface::all() {
        assert_eq!(g.lens(s).unwrap().session_id, "ses_product");
    }
    g.switch(Surface::You);
    let cap = g
        .create_handoff(
            HandoffKind::YouToChat,
            10,
            vec![Claim {
                id: "c".into(),
                text: "shared".into(),
                evidence_tier: EvidenceTier::Cited,
                payload: json!({}),
            }],
            vec![DeliberateExclusion {
                item: "gmail".into(),
                reason: "claim only".into(),
            }],
            json!({"kind": "implementation_campaign"}),
            "user",
        )
        .unwrap();
    assert_eq!(cap.origin_session, "ses_product");
    g.receive_handoff(&cap.id).unwrap();
    assert_eq!(g.session_id(), "ses_product");
    assert!(g
        .lens(Surface::Chat)
        .unwrap()
        .require_connector("gmail")
        .is_err());
}
#[test]
fn all_roles_and_modes_exist() {
    assert_eq!(AgentRole::all().len(), 12);
    assert_eq!(SwarmMode::all().len(), 8);
    assert_eq!(ProjectState::all().len(), 5);
    assert_eq!(Surface::all().len(), 3);
}
