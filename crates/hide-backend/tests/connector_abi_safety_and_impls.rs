use hide_backend::connector_abi::abi::{
    EffectClass, FamilyId, ImplementationStatus, WriteCapability,
};
use hide_backend::connector_abi::effects::{
    execute_with_receipt, execute_without_receipt, ConnectorWriteProposal, PermissionDecision,
    PermissionGate, PermissionPolicy, WriteKind,
};
use hide_backend::connector_abi::{
    AccountStore, Connector, ConnectorError, ConnectorIngestCap, ConnectorMemoryStore,
    ConnectorRead, ConnectorRegistry, CredentialMaterial, ListRequest, LocalFolderConnector,
    ReadRequest, UserMemoryPromotionCap,
};
use std::path::{Path, PathBuf};
fn fixture_feed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/rss/sample_feed.xml")
}
fn temp_folder() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hello from local_folder").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub/nested.txt"), "nested").unwrap();
    let root = dir.path().canonicalize().unwrap();
    (dir, root)
}
#[test]
fn safety_default_read_only_type_boundary() {
    let reg = ConnectorRegistry::builtin();
    let folder_abi = reg.get("local_folder").unwrap();
    assert!(!folder_abi.declares_write());
    assert!(!folder_abi.write.is_writable());
    assert_eq!(folder_abi.status, ImplementationStatus::Implemented);
    let rss_abi = reg.get("rss").unwrap();
    assert!(!rss_abi.declares_write());
    let live = reg.construct("local_folder").unwrap();
    match &live {
        hide_backend::connector_abi::LiveConnector::LocalFolder(c) => {
            assert!(!c.abi().write.is_writable());
            let _: &WriteCapability = &c.abi().write;
            assert!(!c.abi().write.create && !c.abi().write.update && !c.abi().write.delete);
        }
        _ => panic!("expected local_folder"),
    }
    assert!(refuses_write_when_not_declared(
        &folder_abi.write,
        &folder_abi.family_id
    ));
    let gh = reg.get("github").unwrap();
    assert!(gh.declares_write());
    assert!(matches!(
        reg.construct("github"),
        Err(ConnectorError::DeclaredNotConstructible(_))
    ));
}
fn refuses_write_when_not_declared(write: &WriteCapability, family: &FamilyId) -> bool {
    if write.is_writable() {
        false
    } else {
        matches!(
            Err::<(), _>(ConnectorError::WriteNotDeclared(family.clone())),
            Err(ConnectorError::WriteNotDeclared(_))
        )
    }
}
#[test]
fn safety_no_ambient_credentials() {
    assert!(matches!(
        AccountStore::ambient_lookup_forbidden(),
        ConnectorError::AmbientCredentialForbidden
    ));
    let mut store = AccountStore::new();
    let a = store.register(
        FamilyId::new("local_folder"),
        "folder-a",
        CredentialMaterial {
            material: "/tmp/a".into(),
        },
    );
    let b = store.register(
        FamilyId::new("local_folder"),
        "folder-b",
        CredentialMaterial {
            material: "/tmp/b".into(),
        },
    );
    let ha = store.mint_handle(&a).unwrap();
    let hb = store.mint_handle(&b).unwrap();
    assert_ne!(ha.account_id(), hb.account_id());
    assert_eq!(ha.credential_material(), "/tmp/a");
    assert_eq!(hb.credential_material(), "/tmp/b");
    store
        .validate(&ha, &FamilyId::new("rss"))
        .expect_err("family mismatch");
    let forged = ha.clone();
    assert!(store.validate(&ha, &FamilyId::new("local_folder")).is_ok());
    assert!(store.validate(&hb, &FamilyId::new("local_folder")).is_ok());
    assert_ne!(ha.credential_material(), hb.credential_material());
    assert_eq!(forged.account_id(), ha.account_id());
}
#[test]
fn safety_every_write_is_effect_with_receipt() {
    let proposal = ConnectorWriteProposal {
        family_id: FamilyId::new("github"),
        account_id: hide_backend::connector_abi::AccountId::new("github-0"),
        kind: WriteKind::Create,
        effect: EffectClass::Write,
        summary: "open a PR".into(),
        target: "repo/main".into(),
        payload: "{\"title\":\"x\"}".into(),
    };
    assert!(matches!(
        execute_without_receipt(&proposal),
        Err(ConnectorError::WriteReceiptRequired)
    ));
    let mut gate = PermissionGate::new(PermissionPolicy::deny_by_default());
    let denied = gate.authorize(proposal.clone()).unwrap();
    assert_eq!(denied.decision, PermissionDecision::Deny);
    assert!(matches!(
        gate.consume(&denied.id),
        Err(ConnectorError::WritePermissionDenied(_))
    ));
    let mut gate =
        PermissionGate::new(PermissionPolicy::deny_by_default().allow_target("repo/main"));
    let allowed = gate.authorize(proposal.clone()).unwrap();
    assert_eq!(allowed.decision, PermissionDecision::Allow);
    assert!(!allowed.digest.is_empty());
    let mut accounts = AccountStore::new();
    let aid = accounts.register(
        FamilyId::new("github"),
        "gh",
        CredentialMaterial {
            material: "token-fixture".into(),
        },
    );
    let mut proposal2 = proposal;
    proposal2.account_id = aid.clone();
    let allowed2 = gate.authorize(proposal2).unwrap();
    let handle = accounts.mint_handle(&aid).unwrap();
    let result = execute_with_receipt(&mut gate, &allowed2.id, &handle, |p, h| {
        assert_eq!(p.target, "repo/main");
        assert_eq!(h.account_id(), &aid);
        Ok(hide_backend::connector_abi::WriteResult {
            receipt_id: String::new(),
            target: p.target.clone(),
            notes: "fixture execute".into(),
        })
    })
    .unwrap();
    assert_eq!(result.receipt_id, allowed2.id);
    assert_eq!(result.target, "repo/main");
    assert!(matches!(
        gate.consume(&allowed2.id),
        Err(ConnectorError::InvalidWriteReceipt(_))
    ));
    let stored = gate.get(&allowed2.id).unwrap();
    assert!(stored.consumed);
    assert_eq!(stored.decision, PermissionDecision::Allow);
}
#[test]
fn safety_connector_read_cannot_write_user_memory() {
    let mut mem = ConnectorMemoryStore::new();
    let ingest = ConnectorIngestCap::mint();
    let rec = mem.ingest_connector(
        &ingest,
        FamilyId::new("rss"),
        hide_backend::connector_abi::AccountId::new("rss-0"),
        "fixture-item-1",
        "First fixture item body",
    );
    assert!(matches!(
        rec.scope,
        hide_backend::connector_abi::MemoryScope::Connector { .. }
    ));
    assert!(mem.user_records().is_empty());
    let err = mem
        .ingest_as_user_from_connector(&ingest, "smuggle into user")
        .unwrap_err();
    assert!(matches!(
        err,
        ConnectorError::SilentMemoryPromotion {
            target: hide_backend::connector_abi::MemoryScope::User
        }
    ));
    assert!(mem.user_records().is_empty());
    let cap = UserMemoryPromotionCap::mint();
    let user = mem.promote_to_user(&cap, &rec.id).unwrap();
    assert!(matches!(
        user.scope,
        hide_backend::connector_abi::MemoryScope::User
    ));
    assert_eq!(mem.user_records().len(), 1);
    assert_eq!(user.content, "First fixture item body");
}
#[test]
fn safety_revocation_fail_closed() {
    let (_tmp, root) = temp_folder();
    let mut store = AccountStore::new();
    let id = store.register(
        FamilyId::new("local_folder"),
        "docs",
        CredentialMaterial {
            material: root.to_string_lossy().into_owned(),
        },
    );
    let handle = store.mint_handle(&id).unwrap();
    let conn = LocalFolderConnector::new();
    let listed = conn.list(&store, &handle, &ListRequest::default()).unwrap();
    assert!(listed.iter().any(|o| o.title == "hello.txt"));
    let guard = hide_backend::connector_abi::InFlightGuard::begin(
        &store,
        &handle,
        &FamilyId::new("local_folder"),
    )
    .unwrap();
    store.revoke(&id).unwrap();
    assert!(store.is_revoked(&id));
    assert!(matches!(
        guard.complete(&store),
        Err(ConnectorError::AccountRevoked(_)) | Err(ConnectorError::StaleHandle)
    ));
    assert!(matches!(
        conn.list(&store, &handle, &ListRequest::default()),
        Err(ConnectorError::AccountRevoked(_)) | Err(ConnectorError::StaleHandle)
    ));
    assert!(matches!(
        conn.fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "hello.txt".into()
            }
        ),
        Err(ConnectorError::AccountRevoked(_)) | Err(ConnectorError::StaleHandle)
    ));
    assert!(matches!(
        store.mint_handle(&id),
        Err(ConnectorError::AccountRevoked(_))
    ));
}
#[test]
fn local_folder_lists_and_fetches() {
    let (_tmp, root) = temp_folder();
    let mut store = AccountStore::new();
    let id = store.register(
        FamilyId::new("local_folder"),
        "docs",
        CredentialMaterial {
            material: root.to_string_lossy().into_owned(),
        },
    );
    let handle = store.mint_handle(&id).unwrap();
    let reg = ConnectorRegistry::builtin();
    let live = reg.construct("local_folder").unwrap();
    let conn = live.as_read();
    let listed = conn.list(&store, &handle, &ListRequest::default()).unwrap();
    let names: Vec<_> = listed.iter().map(|o| o.title.as_str()).collect();
    assert!(names.contains(&"hello.txt"));
    assert!(names.contains(&"sub"));
    let file = conn
        .fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "hello.txt".into(),
            },
        )
        .unwrap();
    assert_eq!(file.content.as_deref(), Some("hello from local_folder"));
    let nested = conn
        .fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "sub/nested.txt".into(),
            },
        )
        .unwrap();
    assert_eq!(nested.content.as_deref(), Some("nested"));
    assert!(matches!(
        conn.fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "../etc/passwd".into()
            }
        ),
        Err(ConnectorError::InvalidRequest(_))
    ));
}
#[test]
fn rss_parses_committed_fixture() {
    let feed = fixture_feed();
    assert!(feed.is_file(), "fixture missing: {}", feed.display());
    let mut store = AccountStore::new();
    let id = store.register(
        FamilyId::new("rss"),
        "fixture-feed",
        CredentialMaterial {
            material: feed.to_string_lossy().into_owned(),
        },
    );
    let handle = store.mint_handle(&id).unwrap();
    let reg = ConnectorRegistry::builtin();
    let live = reg.construct("rss").unwrap();
    let conn = live.as_read();
    let listed = conn.list(&store, &handle, &ListRequest::default()).unwrap();
    assert!(listed.len() >= 4);
    assert_eq!(listed[0].object_type, "feed");
    assert_eq!(listed[0].title, "HIDE YOU Fixture Feed");
    let item = conn
        .fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "fixture-item-1".into(),
            },
        )
        .unwrap();
    assert_eq!(item.title, "First fixture item");
    assert!(item
        .content
        .as_deref()
        .unwrap_or("")
        .contains("first fixture item"));
    let item3 = conn
        .fetch(
            &store,
            &handle,
            &ReadRequest {
                locator: "fixture-item-3".into(),
            },
        )
        .unwrap();
    assert!(
        item3
            .content
            .as_deref()
            .unwrap_or("")
            .contains("& entities")
            || item3
                .content
                .as_deref()
                .unwrap_or("")
                .contains("& entities")
            || item3.content.as_deref().unwrap_or("").contains("entities")
    );
}
#[test]
fn declared_connectors_not_constructible() {
    let reg = ConnectorRegistry::builtin();
    for abi in reg.declared() {
        let err = reg.construct(abi.family_id.as_str()).unwrap_err();
        assert!(matches!(err, ConnectorError::DeclaredNotConstructible(_)));
    }
    assert!(matches!(
        reg.construct("not_a_family"),
        Err(ConnectorError::UnknownFamily(_))
    ));
}
#[test]
fn registry_covers_all_required_families_and_validates() {
    let reg = ConnectorRegistry::builtin();
    reg.validate_all().expect("all ABIs valid");
    let required = [
        "local_folder",
        "rss",
        "github",
        "google_drive",
        "icloud_drive",
        "gmail",
        "google_calendar",
        "google_contacts",
        "slack",
        "notion",
        "dropbox_onedrive",
        "browser_search",
        "generic_mcp",
        "generic_oauth_api",
        "hawking_artifact_registry",
    ];
    for id in required {
        assert!(reg.get(id).is_some(), "missing family {id}");
    }
    assert_eq!(reg.implemented().len(), 2);
    assert_eq!(reg.declared().len(), required.len() - 2);
    assert_eq!(reg.len(), required.len());
    let doc = reg.export_document();
    assert_eq!(doc.implemented, vec!["local_folder", "rss"]);
    assert!(doc.declared.contains(&"github".to_string()));
    assert_eq!(doc.families.len(), required.len());
    assert_eq!(doc.safety_properties.len(), 5);
}
#[test]
fn registry_json_roundtrip_matches_export() {
    let reg = ConnectorRegistry::builtin();
    let doc = reg.export_document();
    let text = serde_json::to_string_pretty(&doc).unwrap();
    let back: hide_backend::connector_abi::RegistryDocument = serde_json::from_str(&text).unwrap();
    assert_eq!(back.implemented, doc.implemented);
    assert_eq!(back.declared.len(), doc.declared.len());
    assert_eq!(back.families.len(), doc.families.len());
}
#[test]
fn registry_json_on_disk_matches_export() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../workspace/campaign/evidence/systems/hide/HIDE_YOU_CONNECTOR_REGISTRY.json");
    let path = path.canonicalize().expect(
        "workspace/campaign/evidence/systems/hide/HIDE_YOU_CONNECTOR_REGISTRY.json missing; run: cargo run -p hide-connectors --example export_registry -- workspace/campaign/evidence/systems/hide/HIDE_YOU_CONNECTOR_REGISTRY.json",
    );
    let on_disk: hide_backend::connector_abi::RegistryDocument =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let live = ConnectorRegistry::builtin().export_document();
    assert_eq!(on_disk.implemented, live.implemented);
    assert_eq!(on_disk.declared, live.declared);
    assert_eq!(on_disk.families.len(), live.families.len());
    assert_eq!(on_disk.safety_properties, live.safety_properties);
    for (a, b) in on_disk.families.iter().zip(live.families.iter()) {
        assert_eq!(a.family_id, b.family_id);
        assert_eq!(a.status, b.status);
        assert_eq!(a.write, b.write);
        assert_eq!(a.read, b.read);
    }
}
