use hide_kernel::extension_registry::{
    CapabilityKind, CapabilityManifest, Effect, NetworkPolicy, PinSpec, Provenance, Registry,
    ResolveQuery, SandboxReq, SchemaRef, Scope, SecretPolicy,
};
use std::io::Write;
fn tool(id: &str, desc: &str) -> CapabilityManifest {
    let mut m = CapabilityManifest::new(id, "1.0.0", CapabilityKind::Tool, "hide");
    m.description = desc.to_string();
    m
}
fn tool_with_schema(id: &str, desc: &str) -> CapabilityManifest {
    let mut m = tool(id, desc);
    m.input_schema_ref = SchemaRef::with_raw(
        format!("schema://{id}/input"),
        r#"{"type":"object","properties":{"path":{"type":"string"}}}"#,
    );
    m.output_schema_ref =
        SchemaRef::with_raw(format!("schema://{id}/output"), r#"{"type":"string"}"#);
    m.context_cost.schema_tokens = 120;
    m
}
#[test]
fn registers_across_all_kinds() {
    let kinds = [
        CapabilityKind::Tool,
        CapabilityKind::Skill,
        CapabilityKind::Plugin,
        CapabilityKind::Hook,
        CapabilityKind::Mcp,
        CapabilityKind::Acp,
        CapabilityKind::Subagent,
        CapabilityKind::Rule,
        CapabilityKind::Command,
        CapabilityKind::Oracle,
        CapabilityKind::Browser,
        CapabilityKind::Integration,
    ];
    let mut reg = Registry::new();
    for (i, k) in kinds.iter().enumerate() {
        let id = format!("cap.{i}");
        let m = CapabilityManifest::new(&id, "1.0.0", *k, "hide");
        reg.register(m).unwrap();
    }
    assert_eq!(reg.active_len(), kinds.len());
    let idx = reg.index();
    assert_eq!(idx.len(), kinds.len());
    for (i, k) in kinds.iter().enumerate() {
        assert_eq!(reg.kind(&format!("cap.{i}")).unwrap(), *k);
    }
}
#[test]
fn progressive_disclosure_schema_not_loaded_until_requested() {
    let mut reg = Registry::new();
    reg.register(tool_with_schema("fs.read", "read a file"))
        .unwrap();
    let idx = reg.index();
    assert_eq!(idx.len(), 1);
    assert_eq!(idx[0].id, "fs.read");
    let _ranked = reg.resolve_for(&ResolveQuery::new().task("read"));
    assert_eq!(
        reg.schema_load_count(),
        0,
        "index and resolve must never load a schema"
    );
    let full = reg.load_full_schema("fs.read").unwrap();
    assert_eq!(reg.schema_load_count(), 1);
    assert!(full.input.is_some());
    assert!(full.output.is_some());
    assert_eq!(full.input_uri, "schema://fs.read/input");
    assert_eq!(
        full.input.as_ref().unwrap()["properties"]["path"]["type"],
        "string"
    );
    reg.load_full_schema("fs.read").unwrap();
    assert_eq!(reg.schema_load_count(), 2);
}
#[test]
fn bad_schema_surfaces_on_load_only() {
    let mut reg = Registry::new();
    let mut m = tool("bad.schema", "broken schema");
    m.input_schema_ref = SchemaRef::with_raw("schema://bad/input", "{not valid json");
    reg.register(m).unwrap();
    assert_eq!(reg.schema_load_count(), 0);
    let err = reg.load_full_schema("bad.schema").unwrap_err();
    assert!(matches!(
        err,
        hide_kernel::extension_registry::RegistryError::Schema { .. }
    ));
}
#[test]
fn scope_filtering() {
    let mut reg = Registry::new();
    let mut src_only = tool("edit.src", "edit under src");
    src_only.scopes = vec![Scope::Filesystem("src".to_string())];
    reg.register(src_only).unwrap();
    let mut tests_only = tool("edit.tests", "edit under tests");
    tests_only.scopes = vec![Scope::Filesystem("tests".to_string())];
    reg.register(tests_only).unwrap();
    let q = ResolveQuery::new().require_scope(Scope::Filesystem("src/lib.rs".to_string()));
    let ranked = reg.resolve_for(&q);
    let ids: Vec<&str> = ranked.iter().map(|c| c.entry.id.as_str()).collect();
    assert_eq!(ids, vec!["edit.src"]);
    assert!(reg
        .scope_allows(
            "edit.src",
            &Scope::Filesystem("src/deep/mod.rs".to_string())
        )
        .unwrap());
    assert!(!reg
        .scope_allows("edit.tests", &Scope::Filesystem("src/lib.rs".to_string()))
        .unwrap());
}
#[test]
fn role_filtering() {
    let mut reg = Registry::new();
    let mut admin_only = tool("danger.op", "privileged op");
    admin_only.scopes = vec![Scope::Role("admin".to_string())];
    admin_only.effects = vec![Effect::Read, Effect::Privileged];
    reg.register(admin_only).unwrap();
    let general = tool("safe.op", "general op");
    reg.register(general).unwrap();
    let reviewer = reg.resolve_for(&ResolveQuery::new().role("reviewer"));
    let ids: Vec<&str> = reviewer.iter().map(|c| c.entry.id.as_str()).collect();
    assert_eq!(ids, vec!["safe.op"]);
    let admin = reg.resolve_for(&ResolveQuery::new().role("admin"));
    let ids: Vec<&str> = admin.iter().map(|c| c.entry.id.as_str()).collect();
    assert!(ids.contains(&"danger.op"));
    assert!(ids.contains(&"safe.op"));
    let danger = admin.iter().find(|c| c.entry.id == "danger.op").unwrap();
    assert!(danger.role_match);
}
#[test]
fn effect_declaration_retrieval_and_sandbox() {
    let mut reg = Registry::new();
    let mut m = tool("shell.exec", "run a shell command");
    m.effects = vec![Effect::Read, Effect::Execute, Effect::Write];
    m.sandbox = SandboxReq::Subprocess;
    reg.register(m).unwrap();
    let effects = reg.declared_effects("shell.exec").unwrap();
    assert_eq!(effects, vec![Effect::Read, Effect::Execute, Effect::Write]);
    assert!(reg.requires_sandbox("shell.exec").unwrap());
    reg.register(tool("noop", "does nothing")).unwrap();
    assert!(!reg.requires_sandbox("noop").unwrap());
}
#[test]
fn executing_capability_without_sandbox_is_rejected() {
    let mut reg = Registry::new();
    let mut m = tool("rogue.exec", "spawns a process in-process");
    m.effects = vec![Effect::Read, Effect::Process];
    let err = reg.register(m).unwrap_err();
    assert!(matches!(
        err,
        hide_kernel::extension_registry::RegistryError::InvalidManifest { id, .. } if id == "rogue.exec"
    ));
    assert!(!reg.contains("rogue.exec"));
    let mut ok = tool("rogue.exec", "spawns a process, isolated");
    ok.effects = vec![Effect::Read, Effect::Process];
    ok.sandbox = SandboxReq::Subprocess;
    reg.register(ok).unwrap();
    assert!(reg.requires_sandbox("rogue.exec").unwrap());
}
#[test]
fn duplicate_id_rejected() {
    let mut reg = Registry::new();
    reg.register(tool("dup", "first")).unwrap();
    let err = reg.register(tool("dup", "second")).unwrap_err();
    assert!(
        matches!( err, hide_kernel::extension_registry::RegistryError::DuplicateId(id) if id == "dup" )
    );
    assert_eq!(reg.active_len(), 1);
}
#[test]
fn revocation_removes_from_index_resolution_and_load() {
    let mut reg = Registry::new();
    reg.register(tool_with_schema("gone", "to be revoked"))
        .unwrap();
    reg.register(tool("stay", "remains")).unwrap();
    reg.revoke("gone").unwrap();
    assert!(reg.is_revoked("gone"));
    assert!(reg.contains("gone"));
    assert!(matches!(
        reg.register(tool("gone", "sneaky")).unwrap_err(),
        hide_kernel::extension_registry::RegistryError::DuplicateId(_)
    ));
    let idx_ids: Vec<String> = reg.index().into_iter().map(|e| e.id).collect();
    assert_eq!(idx_ids, vec!["stay".to_string()]);
    let ranked = reg.resolve_for(&ResolveQuery::new());
    assert!(ranked.iter().all(|c| c.entry.id != "gone"));
    assert!(matches!(
        reg.load_full_schema("gone").unwrap_err(),
        hide_kernel::extension_registry::RegistryError::Revoked(_)
    ));
    assert!(reg.declared_effects("gone").is_err());
    assert!(matches!(
        reg.revoke("never").unwrap_err(),
        hide_kernel::extension_registry::RegistryError::NotFound(_)
    ));
}
#[test]
fn undeclared_effect_rejected() {
    let mut reg = Registry::new();
    let mut net = tool("fetch", "make a request");
    net.network = NetworkPolicy::Any;
    let err = reg.register(net).unwrap_err();
    match err {
        hide_kernel::extension_registry::RegistryError::UndeclaredEffects { id, missing } => {
            assert_eq!(id, "fetch");
            assert_eq!(missing.0, vec![Effect::Network]);
        }
        other => panic!("expected UndeclaredEffects, got {other:?}"),
    }
    let mut net_ok = tool("fetch", "make a request");
    net_ok.network = NetworkPolicy::AllowList(vec!["api.example.com".to_string()]);
    net_ok.effects = vec![Effect::Read, Effect::Network];
    reg.register(net_ok).unwrap();
    assert!(reg.contains("fetch"));
    let mut sec = tool("vault", "read a secret");
    sec.scopes = vec![Scope::Secret("token".to_string())];
    sec.secrets = SecretPolicy::Named(vec!["token".to_string()]);
    let err = reg.register(sec).unwrap_err();
    assert!(matches!(
        err,
        hide_kernel::extension_registry::RegistryError::UndeclaredEffects { missing, .. }
            if missing.0 == vec![Effect::SecretAccess]
    ));
}
#[test]
fn deterministic_ranking() {
    let build = |order: &[&str]| -> Vec<String> {
        let mut reg = Registry::new();
        for id in order {
            let m = match *id {
                "read.a" => {
                    let mut m = tool("read.a", "search the code");
                    m.effects = vec![Effect::Read];
                    m.context_cost.schema_tokens = 200;
                    m
                }
                "exec.b" => {
                    let mut m = tool("exec.b", "search the code");
                    m.effects = vec![Effect::Read, Effect::Execute];
                    m.sandbox = SandboxReq::Subprocess;
                    m.context_cost.schema_tokens = 50;
                    m
                }
                "read.c" => {
                    let mut m = tool("read.c", "search the code");
                    m.effects = vec![Effect::Read];
                    m.context_cost.schema_tokens = 500;
                    m
                }
                _ => unreachable!(),
            };
            reg.register(m).unwrap();
        }
        reg.resolve_for(&ResolveQuery::new().task("search code"))
            .into_iter()
            .map(|c| c.entry.id)
            .collect()
    };
    let expected = vec![
        "read.a".to_string(),
        "read.c".to_string(),
        "exec.b".to_string(),
    ];
    assert_eq!(build(&["read.a", "exec.b", "read.c"]), expected);
    assert_eq!(build(&["exec.b", "read.c", "read.a"]), expected);
    assert_eq!(build(&["read.c", "read.a", "exec.b"]), expected);
}
#[test]
fn task_relevance_beats_least_privilege() {
    let mut reg = Registry::new();
    let mut relevant = tool("http.get", "fetch a url over http");
    relevant.effects = vec![Effect::Read, Effect::Network];
    relevant.network = NetworkPolicy::Any;
    reg.register(relevant).unwrap();
    reg.register(tool("noop", "unrelated helper")).unwrap();
    let ranked = reg.resolve_for(&ResolveQuery::new().task("fetch url"));
    assert_eq!(ranked[0].entry.id, "http.get");
    assert!(ranked[0].task_matches >= 2);
}
#[test]
fn version_and_provenance_pinning() {
    let mut reg = Registry::new();
    reg.pin(
        "pinned.tool",
        PinSpec {
            version: Some("2.0.0".to_string()),
            source: Some("github.com/hide/pinned".to_string()),
            commit: Some("abc123".to_string()),
        },
    )
    .unwrap();
    let mut wrong = CapabilityManifest::new("pinned.tool", "1.0.0", CapabilityKind::Tool, "hide");
    wrong.provenance = Provenance {
        source: "github.com/hide/pinned".to_string(),
        commit: Some("abc123".to_string()),
        license: "MIT".to_string(),
    };
    assert!(matches!(
        reg.register(wrong).unwrap_err(),
        hide_kernel::extension_registry::RegistryError::PinViolation { .. }
    ));
    let mut right = CapabilityManifest::new("pinned.tool", "2.0.0", CapabilityKind::Tool, "hide");
    right.provenance = Provenance {
        source: "github.com/hide/pinned".to_string(),
        commit: Some("abc123".to_string()),
        license: "MIT".to_string(),
    };
    reg.register(right).unwrap();
    assert_eq!(reg.version("pinned.tool").unwrap(), "2.0.0");
    assert_eq!(
        reg.provenance("pinned.tool").unwrap().commit.as_deref(),
        Some("abc123")
    );
    reg.register(tool("live", "already here")).unwrap();
    assert!(matches!(
        reg.pin(
            "live",
            PinSpec {
                version: Some("9.9.9".to_string()),
                ..Default::default()
            }
        )
        .unwrap_err(),
        hide_kernel::extension_registry::RegistryError::PinViolation { .. }
    ));
}
#[test]
fn empty_identity_rejected() {
    let mut reg = Registry::new();
    let empty_id = CapabilityManifest::new("", "1.0.0", CapabilityKind::Tool, "hide");
    assert!(matches!(
        reg.register(empty_id).unwrap_err(),
        hide_kernel::extension_registry::RegistryError::InvalidManifest { .. }
    ));
    let empty_ver = CapabilityManifest::new("x", "", CapabilityKind::Tool, "hide");
    assert!(matches!(
        reg.register(empty_ver).unwrap_err(),
        hide_kernel::extension_registry::RegistryError::InvalidManifest { .. }
    ));
}
#[test]
fn manifests_load_from_tempdir_json_fixtures() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = tool_with_schema("fs.read", "read a file from the repo");
    a.scopes = vec![Scope::Repo];
    let mut b = CapabilityManifest::new("web.fetch", "1.2.0", CapabilityKind::Integration, "hide");
    b.description = "fetch a page over the network".to_string();
    b.effects = vec![Effect::Read, Effect::Network];
    b.network = NetworkPolicy::AllowList(vec!["example.com".to_string()]);
    b.scopes = vec![Scope::Network("example.com".to_string())];
    for m in [&a, &b] {
        let path = dir.path().join(format!("{}.json", m.id));
        let json = serde_json::to_string_pretty(m).unwrap();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
    }
    let mut files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    let mut reg = Registry::new();
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let m: CapabilityManifest = serde_json::from_str(&text).unwrap();
        reg.register(m).unwrap();
    }
    assert_eq!(reg.active_len(), 2);
    assert!(reg
        .declared_effects("web.fetch")
        .unwrap()
        .contains(&Effect::Network));
    assert!(reg
        .scope_allows("web.fetch", &Scope::Network("example.com".to_string()))
        .unwrap());
    assert_eq!(reg.schema_load_count(), 0);
    let full = reg.load_full_schema("fs.read").unwrap();
    assert!(full.input.is_some());
}
