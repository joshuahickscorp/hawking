use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn run_git(root: &Path, args: &[&str]) -> Output {
    Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .expect("git must be available for the staging contract test")
}

fn git_ok(root: &Path, args: &[&str]) -> String {
    let output = run_git(root, args);
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn run_hawking(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hawking"))
        .current_dir(root)
        .args(args)
        .output()
        .expect("hawking binary must be runnable")
}

fn temp_root() -> PathBuf {
    std::env::temp_dir().join(format!(
        "hawking_external_worker_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

#[test]
fn stage_preserves_dirty_canonical_state_and_result_is_non_promoting() {
    let root = temp_root();
    let staging = root
        .parent()
        .expect("temporary directory has a parent")
        .join(format!(
            "{}_staging",
            root.file_name().unwrap().to_string_lossy()
        ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("H-MANIFESTO.md"), "manifesto marker\n").unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::write(root.join("tracked.txt"), "canonical base\n").unwrap();
    fs::create_dir_all(root.join("workspace/campaign/odyssey")).unwrap();
    fs::write(
        root.join("workspace/campaign/odyssey/CONTINUATION.json"),
        r#"{
          "objective": "Fixture objective",
          "active_specimen": "fixture",
          "active_workunit": "fixture worker",
          "next_action": "stop at the authority boundary",
          "current_source_correctness_frontier": {"status": "BLOCKED", "last_bounded_frontier": "fixture"},
          "organ_passport_frontier": {"status": "STATIC", "passport_count": 1},
          "native_nr_admission_frontier": {"status": "WITHHELD", "complete_runnable_nr_present": false},
          "complete_nr_accounting_frontier": {"status": "OPEN", "complete_nr_ebpw": null},
          "evidence_refs": []
        }"#,
    )
    .unwrap();
    fs::create_dir_all(root.join("tools/foundry")).unwrap();
    fs::write(
        root.join("tools/foundry/GRAVITY_METHOD_REGISTRY.json"),
        r#"{
          "execution_frontier": {"updated_at": "fixture", "ordering_rule": "cheap falsifier first", "sealed_now": [], "active_native_lane": [], "withheld_without_native_bank": []},
          "organ_passports": {"entries": [{"id": "fixture.passport", "semantic_role": "fixture organ", "recognition": {"all": []}, "trait_projection": {}, "tensor_contract": {}, "state_contract": {}, "source_accounting": {}}]},
          "methods": [{"id": "fixture-method", "title": "Passport fixture method", "family": "fixture", "status": "candidate", "required_controls": ["fixture-control"], "evidence": [], "laws": [], "claim_boundary": "fixture only", "reopen_if": "fixture changes"}]
        }"#,
    )
    .unwrap();

    git_ok(&root, &["init", "-b", "main"]);
    git_ok(&root, &["config", "user.name", "Hawking Test"]);
    git_ok(
        &root,
        &["config", "user.email", "hawking-test@example.invalid"],
    );
    git_ok(&root, &["add", "."]);
    git_ok(&root, &["commit", "-m", "test base"]);
    let base = git_ok(&root, &["rev-parse", "HEAD"]);

    // This dirty state belongs to the canonical checkout and must not be
    // copied, reset, or cleaned by staging.
    fs::write(root.join("tracked.txt"), "canonical dirty change\n").unwrap();
    let before_status = git_ok(&root, &["status", "--porcelain=v1"]);

    let staged = run_hawking(
        &root,
        &[
            "agent",
            "stage",
            "hawking",
            "passport-check",
            "--objective",
            "Inspect one deterministic passport validation gap.",
            "--scope",
            "docs",
            "--staging-root",
            staging.to_str().unwrap(),
            "--json",
        ],
    );
    assert!(
        staged.status.success(),
        "stage failed: {}",
        String::from_utf8_lossy(&staged.stderr)
    );
    let stage_json: Value = serde_json::from_slice(&staged.stdout).unwrap();
    assert_eq!(stage_json["schema"], "hawking.external_worker.stage.v1");
    assert_eq!(stage_json["task_id"], "passport-check");
    assert_eq!(stage_json["source_commit"], base);
    assert_eq!(stage_json["branch"], "worker/passport-check");
    assert_eq!(stage_json["canonical_worktree_dirty_at_creation"], true);
    assert_eq!(stage_json["allowed_scope"][0], "docs");
    assert_eq!(stage_json["worktree_registered"], true);
    assert_eq!(stage_json["promotion_allowed"], Value::Null);
    assert_eq!(
        stage_json["objective"],
        "Inspect one deterministic passport validation gap."
    );
    assert_eq!(git_ok(&root, &["status", "--porcelain=v1"]), before_status);

    let worktree = PathBuf::from(stage_json["worktree_path"].as_str().unwrap());
    assert_eq!(
        fs::read_to_string(worktree.join("tracked.txt")).unwrap(),
        "canonical base\n"
    );
    assert_eq!(git_ok(&worktree, &["status", "--porcelain=v1"]), "");
    let packet = fs::read_to_string(stage_json["task_packet_path"].as_str().unwrap()).unwrap();
    for section in [
        "OBJECTIVE",
        "SOURCE COMMIT",
        "WORKTREE",
        "ALLOWED ACTIONS",
        "FORBIDDEN ACTIONS",
        "AUTHORITATIVE INPUTS",
        "EXPECTED OUTPUT",
        "TESTS",
        "CLAIM BOUNDARY",
        "STOP CONDITIONS",
    ] {
        assert!(packet.contains(section), "packet lacks {section}");
    }

    let context = run_hawking(
        &root,
        &[
            "agent",
            "context",
            "hawking",
            "passport-check",
            "--staging-root",
            staging.to_str().unwrap(),
            "--json",
        ],
    );
    assert!(
        context.status.success(),
        "context failed: {}",
        String::from_utf8_lossy(&context.stderr)
    );
    let context_json: Value = serde_json::from_slice(&context.stdout).unwrap();
    assert_eq!(context_json["schema"], "hawking.external_worker.context.v1");
    assert_eq!(context_json["identity"]["task_id"], "passport-check");
    assert_eq!(context_json["identity"]["staging_base_revision"], base);
    assert_eq!(context_json["authority"]["promotion_allowed"], false);
    assert!(context_json["authority"]["forbidden_actions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap().contains("GPU")));
    assert_eq!(
        context_json["organ_knowledge"]["entries"][0]["id"],
        "fixture.passport"
    );
    assert_eq!(
        context_json["gravity_knowledge"]["methods"][0]["id"],
        "fixture-method"
    );
    let context_json_path = staging.join("packets/passport-check.context.json");
    let context_markdown_path = staging.join("packets/passport-check.context.md");
    assert!(context_json_path.is_file());
    assert!(context_markdown_path.is_file());
    assert!(fs::read_to_string(&context_markdown_path)
        .unwrap()
        .contains("CURRENT FRONTIER"));
    assert_eq!(git_ok(&root, &["status", "--porcelain=v1"]), before_status);
    assert_eq!(git_ok(&worktree, &["status", "--porcelain=v1"]), "");

    let implicit_reuse = run_hawking(
        &root,
        &[
            "agent",
            "stage",
            "hawking",
            "passport-check",
            "--staging-root",
            staging.to_str().unwrap(),
        ],
    );
    assert!(!implicit_reuse.status.success());
    assert!(String::from_utf8_lossy(&implicit_reuse.stderr).contains("implicit staging reuse"));

    let explicit_reuse = run_hawking(
        &root,
        &[
            "agent",
            "stage",
            "hawking",
            "passport-check",
            "--staging-root",
            staging.to_str().unwrap(),
            "--reuse",
            "--json",
        ],
    );
    assert!(
        explicit_reuse.status.success(),
        "reuse failed: {}",
        String::from_utf8_lossy(&explicit_reuse.stderr)
    );
    let reuse_json: Value = serde_json::from_slice(&explicit_reuse.stdout).unwrap();
    assert_eq!(reuse_json["reused"], true);

    fs::create_dir_all(worktree.join(".hawking-worker")).unwrap();
    fs::write(
        worktree.join(".hawking-worker/result.json"),
        r#"{
          "schema": "hawking.external_worker.worker_report.v1",
          "tests": [{"command": "cargo check --workspace", "status": "PASS"}],
          "claims_supported": ["bounded test evidence"]
        }"#,
    )
    .unwrap();
    fs::write(worktree.join("tracked.txt"), "worker change\n").unwrap();

    let observed = run_hawking(
        &root,
        &[
            "agent",
            "result",
            "passport-check",
            "--staging-root",
            staging.to_str().unwrap(),
            "--json",
        ],
    );
    assert!(
        observed.status.success(),
        "result failed: {}",
        String::from_utf8_lossy(&observed.stderr)
    );
    let result_json: Value = serde_json::from_slice(&observed.stdout).unwrap();
    assert_eq!(result_json["schema"], "hawking.external_worker.result.v1");
    assert_eq!(result_json["base_commit"], base);
    assert_eq!(result_json["resulting_commit"], Value::Null);
    assert_eq!(result_json["promotion_allowed"], false);
    assert_eq!(result_json["tests_or_checks"][0]["status"], "PASS");
    assert!(result_json["changed_files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "tracked.txt"));
    assert!(result_json["remaining_risks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap().contains("does not execute")));

    git_ok(
        &worktree,
        &["add", "tracked.txt", ".hawking-worker/result.json"],
    );
    git_ok(&worktree, &["commit", "-m", "worker result"]);
    let committed = run_hawking(
        &root,
        &[
            "agent",
            "result",
            "passport-check",
            "--staging-root",
            staging.to_str().unwrap(),
            "--json",
        ],
    );
    assert!(committed.status.success());
    let committed_json: Value = serde_json::from_slice(&committed.stdout).unwrap();
    assert!(committed_json["resulting_commit"].is_string());
    assert_eq!(committed_json["worktree_status"], "clean");
    assert_eq!(git_ok(&root, &["status", "--porcelain=v1"]), before_status);

    let _ = run_git(
        &root,
        &["worktree", "remove", "--force", worktree.to_str().unwrap()],
    );
    let _ = fs::remove_dir_all(&staging);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn staging_root_inside_canonical_checkout_is_rejected() {
    let root = temp_root();
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("H-MANIFESTO.md"), "manifesto marker\n").unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::write(root.join("tracked.txt"), "base\n").unwrap();
    git_ok(&root, &["init", "-b", "main"]);
    git_ok(&root, &["config", "user.name", "Hawking Test"]);
    git_ok(
        &root,
        &["config", "user.email", "hawking-test@example.invalid"],
    );
    git_ok(&root, &["add", "."]);
    git_ok(&root, &["commit", "-m", "test base"]);

    let inside = root.join("staging");
    let output = run_hawking(
        &root,
        &[
            "agent",
            "stage",
            "hawking",
            "unsafe-path",
            "--staging-root",
            inside.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside the canonical checkout"));
    assert!(!inside.exists());
    let _ = fs::remove_dir_all(&root);
}
