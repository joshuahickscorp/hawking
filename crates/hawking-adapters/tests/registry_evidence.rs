use hawking_adapters::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, ProviderAvailability,
};
use hawking_adapters::evidence::{validate_family_evidence, workspace_root};
use hawking_adapters::registry::builtin_registry;
use hawking_adapters::support_level::SupportLevel;
use hawking_adapters::FamilyDescriptor;
fn empty_abi() -> FamilyAbi {
    FamilyAbi {
        source_config_classes: AbiListField::null("test"),
        tensor_namespace_rules: AbiField::null("test"),
        tokenizer: AbiField::null("test"),
        chat_template: AbiField::null("test"),
        attention_or_state: AbiField::null("test"),
        topology: AbiField::null("test"),
        normalization: AbiField::null("test"),
        positional_encoding: AbiField::null("test"),
        kv_or_state_format: AbiField::null("test"),
        context_limits: ContextLimits::unknown("test"),
        artifact_codecs: AbiListField::null("test"),
        providers: ProviderAvailability {
            cpu: AbiField::null("test"),
            metal: AbiField::null("test"),
            cuda: AbiField::null("test"),
        },
        fabric_partition_boundaries: AbiField::null("test"),
        capability_limitations: &[],
        source_precision_classes: AbiListField::null("test"),
    }
}
#[test]
fn every_family_evidence_present() {
    let r = builtin_registry();
    r.validate_all_evidence()
        .unwrap_or_else(|errs| panic!("evidence failures:\n{}", errs.join("\n")));
}
#[test]
fn every_family_abi_fields_complete() {
    let r = builtin_registry();
    r.validate_all_abi()
        .unwrap_or_else(|errs| panic!("ABI incomplete:\n{}", errs.join("\n")));
}
#[test]
fn level_without_evidence_fails() {
    let root = workspace_root();
    let bogus = FamilyDescriptor {
        id: "bogus",
        aliases: &[],
        display_name: "Bogus",
        level: SupportLevel::SyntheticParity,
        evidence: &[], // empty — must fail
        module: "nowhere",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &bogus).unwrap_err();
    assert!(
        err.contains("requires at least one named evidence"),
        "got: {err}"
    );
}
#[test]
fn level_with_wrong_evidence_kind_fails() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "Cargo.toml",
        claim: "exists but wrong kind",
        kind: EvidenceKind::Description,
    }];
    let d = FamilyDescriptor {
        id: "wrong_kind",
        aliases: &[],
        display_name: "Wrong",
        level: SupportLevel::SmallRealCheckpoint,
        evidence: EV,
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(err.contains("requires evidence kind") && err.contains("small_checkpoint_run"));
}
#[test]
fn source_header_grade_requires_source_header_kind() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "Cargo.toml",
        claim: "file exists but is not a parsed source header",
        kind: EvidenceKind::Description,
    }];
    let d = FamilyDescriptor {
        id: "header_claim",
        aliases: &[],
        display_name: "Header",
        level: SupportLevel::SourceHeaderValidated,
        evidence: EV,
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(err.contains("source_header"), "got: {err}");
}
#[test]
fn real_tensor_decode_grade_requires_matching_kind() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "Cargo.toml",
        claim: "wrong kind for REAL_TENSOR_DECODE",
        kind: EvidenceKind::SyntheticParity,
    }];
    let d = FamilyDescriptor {
        id: "tensor_claim",
        aliases: &[],
        display_name: "Tensor",
        level: SupportLevel::RealTensorDecode,
        evidence: EV,
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(err.contains("real_tensor_decode"), "got: {err}");
}
#[test]
fn production_always_fails() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "Cargo.toml",
        claim: "exists but PRODUCTION is still forbidden",
        kind: EvidenceKind::ProductionReceipt,
    }];
    let prod = FamilyDescriptor {
        id: "fake_prod",
        aliases: &[],
        display_name: "Fake",
        level: SupportLevel::Production,
        evidence: EV,
        module: "x",
        executes: true,
        serve_registered: true,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &prod).unwrap_err();
    assert!(err.contains("PRODUCTION"), "got: {err}");
}
#[test]
fn missing_evidence_path_fails() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "this/path/does/not/exist.rs",
        claim: "phantom",
        kind: EvidenceKind::SmallCheckpointRun,
    }];
    let d = FamilyDescriptor {
        id: "missing",
        aliases: &[],
        display_name: "Missing",
        level: SupportLevel::SmallRealCheckpoint,
        evidence: EV,
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi: empty_abi(),
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(err.contains("does not exist"), "got: {err}");
}
#[test]
fn incomplete_abi_field_fails() {
    let root = workspace_root();
    let mut abi = empty_abi();
    abi.tokenizer = AbiField {
        value: None,
        null_reason: None,
    };
    let d = FamilyDescriptor {
        id: "incomplete",
        aliases: &[],
        display_name: "Incomplete",
        level: SupportLevel::Declared,
        evidence: &[],
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
        abi,
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(
        err.contains("tokenizer") || err.contains("ABI field"),
        "got: {err}"
    );
}
#[test]
fn no_family_is_production_in_builtin() {
    let r = builtin_registry();
    for d in r.families() {
        assert_ne!(d.level, SupportLevel::Production, "{}", d.id);
    }
}
