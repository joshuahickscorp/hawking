//! Deterministic, model-free tests for the verification plane.

use hide_verify::{
    apply_gate, all_profiles, invalidated_ids, invalidated_receipts, paths_intersect, profile_for,
    probabilistic_can_override_deterministic, source_hash, CheckKind, GateDecision, Oracle,
    OracleClass, ReviewRole, ReviewRoleProfile, Severity, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationInput, VerificationReceipt, VerificationTier, Verdict,
};

/// Build the "dirty" fixture with a known layout so line numbers are exact:
///   line 1   : `fn process(...) {`     (long-function signature)
///   line 2   : `... .unwrap();`         (unwrap outside test)
///   3..=202  : filler                   (pushes the body over the threshold)
///   line 203 : `// ... em dash ...`     (house-rule dash violation)
///   line 204 : `todo!();`               (panic-family marker)
///   line 205 : return
///   line 206 : `}`                      (closes the long function)
fn dirty_fixture() -> String {
    let mut s = String::new();
    s.push_str("fn process(data: &str) -> String {\n"); // 1
    s.push_str("    let value = data.parse::<i32>().unwrap();\n"); // 2
    for _ in 0..200 {
        s.push_str("    let _ = value;\n"); // 3..=202
    }
    // Line 203 carries an em dash (U+2014), written as an escape so this test
    // file itself contains no banned character.
    s.push_str(&format!("    // note {} details\n", '\u{2014}')); // 203
    s.push_str("    todo!();\n"); // 204
    s.push_str("    String::new()\n"); // 205
    s.push_str("}\n"); // 206
    s
}

fn clean_fixture() -> &'static str {
    "pub fn add(a: i32, b: i32) -> i32 {\n\
     \x20   a + b\n\
     }\n\
     \n\
     #[cfg(test)]\n\
     mod tests {\n\
     \x20   use super::*;\n\
     \x20   #[test]\n\
     \x20   fn adds() {\n\
     \x20       assert_eq!(add(2, 2), 4);\n\
     \x20       let v: Option<i32> = Some(1);\n\
     \x20       let _ = v.unwrap();\n\
     \x20   }\n\
     }\n"
}

fn find<'a>(findings: &'a [hide_verify::Finding], check: CheckKind) -> Vec<&'a hide_verify::Finding> {
    findings.iter().filter(|f| f.check == check).collect()
}

#[test]
fn static_analysis_flags_dirty_fixture_with_correct_line_and_severity() {
    let oracle = StaticAnalysisOracle::new();
    let findings = oracle.analyze_source("process.rs", &dirty_fixture());

    // (a) unwrap outside test -> line 2, Warning.
    let unwraps = find(&findings, CheckKind::UnwrapOutsideTest);
    assert_eq!(unwraps.len(), 1, "expected exactly one unwrap finding");
    assert_eq!(unwraps[0].line, 2);
    assert_eq!(unwraps[0].severity, Severity::Warning);
    assert_eq!(unwraps[0].file, "process.rs");

    // (c) em dash -> line 203, Error.
    let dashes = find(&findings, CheckKind::EmDash);
    assert_eq!(dashes.len(), 1, "expected exactly one dash finding");
    assert_eq!(dashes[0].line, 203);
    assert_eq!(dashes[0].severity, Severity::Error);

    // (b) todo! marker -> line 204, Error.
    let markers = find(&findings, CheckKind::PanicMarker);
    assert_eq!(markers.len(), 1, "expected exactly one marker finding");
    assert_eq!(markers[0].line, 204);
    assert_eq!(markers[0].severity, Severity::Error);

    // (d) long function -> signature line 1, Warning.
    let longs = find(&findings, CheckKind::LongFunction);
    assert_eq!(longs.len(), 1, "expected exactly one long-function finding");
    assert_eq!(longs[0].line, 1);
    assert_eq!(longs[0].severity, Severity::Warning);
}

#[test]
fn static_analysis_clean_fixture_yields_pass() {
    let oracle = StaticAnalysisOracle::new();
    let findings = oracle.analyze_source("lib.rs", clean_fixture());
    assert!(
        findings.is_empty(),
        "clean fixture must have no findings, got {findings:?}"
    );

    let input = VerificationInput::from_sources(vec![SourceFile::new("lib.rs", clean_fixture())]);
    let outcome = oracle.evaluate(&input);
    assert_eq!(outcome.verdict, Verdict::Pass);
    assert_eq!(oracle.tier(), VerificationTier::Tier1Deterministic);
    assert_eq!(oracle.class(), OracleClass::Deterministic);
}

#[test]
fn dirty_fixture_evaluates_to_fail() {
    let oracle = StaticAnalysisOracle::new();
    let input = VerificationInput::from_sources(vec![SourceFile::new("process.rs", dirty_fixture())]);
    let outcome = oracle.evaluate(&input);
    assert!(outcome.verdict.is_fail());
    // Every blocking finding shows up as a reason.
    assert!(!outcome.verdict.reasons().is_empty());
    assert!(outcome.evidence.findings.len() >= 4);
}

#[test]
fn dash_check_catches_both_en_and_em() {
    let oracle = StaticAnalysisOracle::new();
    // Build a line containing an en dash (U+2013) and an em dash (U+2014) via
    // escapes, so the fixture, not this file, carries the characters.
    let src = format!("// en {} then em {} end\n", '\u{2013}', '\u{2014}');
    let findings = oracle.analyze_source("x.rs", &src);
    let dashes = find(&findings, CheckKind::EmDash);
    assert_eq!(dashes.len(), 2, "must catch both U+2013 and U+2014");
    assert!(dashes.iter().all(|f| f.line == 1));
    assert!(dashes.iter().all(|f| f.severity == Severity::Error));
    assert!(dashes.iter().any(|f| f.message.contains("2013")));
    assert!(dashes.iter().any(|f| f.message.contains("2014")));
}

#[test]
fn unwrap_inside_test_is_not_flagged_but_outside_is() {
    let oracle = StaticAnalysisOracle::new();
    let src = "fn live() {\n\
               \x20   let x: Option<i32> = None;\n\
               \x20   let _ = x.unwrap();\n\
               }\n\
               #[cfg(test)]\n\
               mod tests {\n\
               \x20   #[test]\n\
               \x20   fn t() {\n\
               \x20       let y: Option<i32> = Some(1);\n\
               \x20       let _ = y.unwrap();\n\
               \x20   }\n\
               }\n";
    let findings = oracle.analyze_source("m.rs", src);
    let unwraps = find(&findings, CheckKind::UnwrapOutsideTest);
    assert_eq!(unwraps.len(), 1, "only the non-test unwrap is flagged");
    assert_eq!(unwraps[0].line, 3);
}

#[test]
fn static_analysis_walks_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn f() {\n    let x: Option<i32> = None;\n    let _ = x.unwrap();\n}\n").unwrap();
    std::fs::write(dir.path().join("notes.txt"), "unwrap() everywhere").unwrap();

    let oracle = StaticAnalysisOracle::new();
    let findings = oracle.analyze_dir(dir.path()).unwrap();
    // The .txt file is ignored; only the .rs unwrap is flagged.
    let unwraps = find(&findings, CheckKind::UnwrapOutsideTest);
    assert_eq!(unwraps.len(), 1);
}

fn receipt(id: &str, scope: &[&str]) -> VerificationReceipt {
    VerificationReceipt::new(
        id,
        VerificationTier::Tier1Deterministic,
        "static_analysis",
        None,
        scope.iter().map(|s| s.to_string()).collect(),
        source_hash(b"snapshot"),
        Verdict::Pass,
        0,
        1,
    )
}

#[test]
fn rereview_invalidates_exactly_intersecting_receipts() {
    let receipts = vec![
        receipt("r1", &["crates/a/src/lib.rs"]),
        receipt("r2", &["crates/b/src/mod.rs"]),
        receipt("r3", &["crates/a/src"]), // directory scope
        receipt("r4", &["crates/a/src/other.rs"]),
    ];
    let changed = vec!["crates/a/src/lib.rs".to_string()];

    let invalidated = invalidated_ids(&receipts, &changed);
    // r1 (exact) and r3 (directory contains the file) are invalidated; r2 and r4
    // are disjoint and stay valid.
    assert_eq!(invalidated, vec!["r1".to_string(), "r3".to_string()]);

    let refs = invalidated_receipts(&receipts, &changed);
    assert_eq!(refs.len(), 2);
    assert!(refs.iter().all(|r| r.verification_id == "r1" || r.verification_id == "r3"));
}

#[test]
fn path_intersection_rules() {
    assert!(paths_intersect("a/b.rs", "a/b.rs"));
    assert!(paths_intersect("a", "a/b.rs")); // dir contains file
    assert!(paths_intersect("a/b.rs", "a")); // symmetric
    assert!(paths_intersect("./a/b.rs", "a/b.rs")); // normalized
    assert!(!paths_intersect("a/b.rs", "a/bc.rs")); // sibling, not a prefix
    assert!(!paths_intersect("ab", "a/b.rs")); // partial component is not containment
}

#[test]
fn selecting_a_review_role_returns_a_profile_not_a_verdict() {
    // A Tier4 review-role selection is pure data: it returns a profile and makes
    // no model call. The compiler enforces the type: `profile_for` returns a
    // `ReviewRoleProfile`, never a `Verdict`.
    let profile: ReviewRoleProfile = profile_for(ReviewRole::Security);
    assert_eq!(profile.role, ReviewRole::Security);
    assert!(!profile.focus.is_empty());
    assert!(!profile.context_kinds.is_empty());
    assert_eq!(profile.output_schema_ref, "hide.review.security.v1");
    assert!(!profile.acceptance.is_empty());

    // All eight roles are present, each with a distinct schema ref.
    let profiles = all_profiles();
    assert_eq!(profiles.len(), 8);
    let mut refs: Vec<String> = profiles.iter().map(|p| p.output_schema_ref.clone()).collect();
    refs.sort();
    refs.dedup();
    assert_eq!(refs.len(), 8, "each role has a unique output schema ref");
}

#[test]
fn verification_receipt_round_trips_with_stable_shape() {
    let original = VerificationReceipt::new(
        "vfy-001",
        VerificationTier::Tier1Deterministic,
        "cargo_test",
        Some("cargo test -p hide-verify".to_string()),
        vec!["crates/hide-verify/src/lib.rs".to_string()],
        source_hash(b"the source"),
        Verdict::Fail {
            reasons: vec!["1 test failed".to_string()],
        },
        1_000,
        250,
    );

    let json = original.to_json().unwrap();
    let parsed = VerificationReceipt::from_json(&json).unwrap();
    assert_eq!(original, parsed);

    // Stable shape: the expected keys are present, the tier and verdict use their
    // documented tags, and Option/command is present.
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    for key in [
        "verification_id",
        "tier",
        "oracle",
        "command",
        "scope",
        "source_hash",
        "verdict",
        "started_ms",
        "duration_ms",
    ] {
        assert!(value.get(key).is_some(), "receipt missing key {key}");
    }
    assert_eq!(value["tier"], "tier1_deterministic");
    assert_eq!(value["verdict"]["status"], "fail");
    assert_eq!(value["verdict"]["reasons"][0], "1 test failed");
}

#[test]
fn verdict_shapes_serialize_stably() {
    assert_eq!(
        serde_json::to_string(&Verdict::Pass).unwrap(),
        r#"{"status":"pass"}"#
    );
    assert_eq!(
        serde_json::to_string(&Verdict::Skipped {
            why: "no changes".to_string()
        })
        .unwrap(),
        r#"{"status":"skipped","why":"no changes"}"#
    );
}

#[test]
fn deterministic_fail_outranks_probabilistic_pass() {
    // THE authority rule: a Tier4 review Pass must NOT rescue a Tier1
    // deterministic Fail.
    let verdicts = vec![
        TieredVerdict::new(
            VerificationTier::Tier1Deterministic,
            "cargo_test",
            Verdict::Fail {
                reasons: vec!["test_foo failed".to_string()],
            },
        ),
        TieredVerdict::new(
            VerificationTier::Tier4Review,
            "correctness_reviewer",
            Verdict::Pass,
        ),
    ];
    match apply_gate(&verdicts) {
        GateDecision::Reject { reasons } => {
            assert!(reasons.iter().any(|r| r.contains("test_foo failed")));
        }
        other => panic!("review Pass must not override a deterministic Fail; got {other:?}"),
    }

    // The invariant is encoded as a value too.
    assert!(!probabilistic_can_override_deterministic());
}

#[test]
fn gate_accepts_when_deterministic_passes_and_review_clean() {
    let verdicts = vec![
        TieredVerdict::new(
            VerificationTier::Tier1Deterministic,
            "static_analysis",
            Verdict::Pass,
        ),
        TieredVerdict::new(VerificationTier::Tier4Review, "scope_reviewer", Verdict::Pass),
    ];
    assert_eq!(apply_gate(&verdicts), GateDecision::Accept);
}

#[test]
fn gate_is_inconclusive_without_a_deterministic_pass() {
    // A review alone can never carry a change to Accept.
    let verdicts = vec![TieredVerdict::new(
        VerificationTier::Tier4Review,
        "correctness_reviewer",
        Verdict::Pass,
    )];
    assert_eq!(apply_gate(&verdicts), GateDecision::Inconclusive);
}

#[test]
fn review_fail_blocks_even_after_deterministic_pass() {
    let verdicts = vec![
        TieredVerdict::new(
            VerificationTier::Tier1Deterministic,
            "cargo_build",
            Verdict::Pass,
        ),
        TieredVerdict::new(
            VerificationTier::Tier4Review,
            "security_reviewer",
            Verdict::Fail {
                reasons: vec!["hardcoded secret".to_string()],
            },
        ),
    ];
    assert!(matches!(apply_gate(&verdicts), GateDecision::Reject { .. }));
}
