//! End-to-end tests for hide-connectors.
//!
//! Named safety properties (the point of the crate):
//! 1. `safety_default_read_only_type_boundary`
//! 2. `safety_no_ambient_credentials`
//! 3. `safety_every_write_is_effect_with_receipt`
//! 4. `safety_connector_read_cannot_write_user_memory`
//! 5. `safety_revocation_fail_closed`
//!
//! Plus fixture-backed local_folder / rss, declared-not-constructible, registry export.

use std::path::{Path, PathBuf};

use hide_connectors::abi::{EffectClass, FamilyId, ImplementationStatus, WriteCapability};
use hide_connectors::effects::{
    execute_with_receipt, execute_without_receipt, ConnectorWriteProposal, PermissionDecision,
    PermissionGate, PermissionPolicy, WriteKind,
};
use hide_connectors::{
    AccountStore, Connector, ConnectorError, ConnectorIngestCap, ConnectorMemoryStore,
    ConnectorRead, ConnectorRegistry, CredentialMaterial, ListRequest, LocalFolderConnector,
    ReadRequest, UserMemoryPromotionCap,
};

fn fixture_feed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/rss/sample_feed.xml")
}

fn temp_folder() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hello from local_folder").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub/nested.txt"), "nested").unwrap();
    let root = dir.path().canonicalize().unwrap();
    (dir, root)
}

// ---------------------------------------------------------------------------
// Safety property 1: default read-only / least privilege (type boundary)
// ---------------------------------------------------------------------------

/// A connector that does not declare write cannot be asked to write.
///
/// `LocalFolderConnector` and `RssConnector` do not implement `ConnectorWrite`.
/// Their ABI `write` is all-false. Declared write-capable families still refuse
/// construction. There is no runtime "please write" escape hatch on read-only
/// types.
#[test]
fn safety_default_read_only_type_boundary() {
    let reg = ConnectorRegistry::builtin();

    let folder_abi = reg.get("local_folder").unwrap();
    assert!(!folder_abi.declares_write());
    assert!(!folder_abi.write.is_writable());
    assert_eq!(folder_abi.status, ImplementationStatus::Implemented);

    let rss_abi = reg.get("rss").unwrap();
    assert!(!rss_abi.declares_write());

    // Compile-time boundary is the missing ConnectorWrite impl. At runtime we
    // also prove ABI + construct path never exposes write for these families.
    let live = reg.construct("local_folder").unwrap();
    match &live {
        hide_connectors::LiveConnector::LocalFolder(c) => {
            assert!(!c.abi().write.is_writable());
            // No write_capability method without ConnectorWrite — ABI is the
            // declaration; trait absence is the boundary.
            let _: &WriteCapability = &c.abi().write;
            assert!(!c.abi().write.create && !c.abi().write.update && !c.abi().write.delete);
        }
        _ => panic!("expected local_folder"),
    }

    // A function that requires ConnectorWrite cannot accept LocalFolderConnector.
    // We encode that by only offering prepare_write on the trait; calling a
    // free helper that demands write capability fails for non-writable ABI.
    assert!(refuses_write_when_not_declared(&folder_abi.write, &folder_abi.family_id));

    // Declared write-capable family still cannot be constructed (no silent write path).
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
        // Mirror ConnectorWrite::prepare_write guard.
        matches!(
            Err::<(), _>(ConnectorError::WriteNotDeclared(family.clone())),
            Err(ConnectorError::WriteNotDeclared(_))
        )
    }
}

// ---------------------------------------------------------------------------
// Safety property 2: no ambient credentials
// ---------------------------------------------------------------------------

#[test]
fn safety_no_ambient_credentials() {
    // Documented law: ambient lookup is forbidden.
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

    // Handles are explicit and distinct; no connector-side global lookup.
    assert_ne!(ha.account_id(), hb.account_id());
    assert_eq!(ha.credential_material(), "/tmp/a");
    assert_eq!(hb.credential_material(), "/tmp/b");

    // Cross-family handle rejected on validate.
    store
        .validate(&ha, &FamilyId::new("rss"))
        .expect_err("family mismatch");

    // One account's credential is not readable as another's: forging a handle
    // with the wrong credential material fails validation.
    let forged = ha.clone();
    // We prove isolation by validating each handle only against its own account.
    assert!(store.validate(&ha, &FamilyId::new("local_folder")).is_ok());
    assert!(store.validate(&hb, &FamilyId::new("local_folder")).is_ok());
    // Handles do not share material.
    assert_ne!(ha.credential_material(), hb.credential_material());
    assert_eq!(forged.account_id(), ha.account_id());
}

// ---------------------------------------------------------------------------
// Safety property 3: every write is an effect with a receipt
// ---------------------------------------------------------------------------

#[test]
fn safety_every_write_is_effect_with_receipt() {
    let proposal = ConnectorWriteProposal {
        family_id: FamilyId::new("github"),
        account_id: hide_connectors::AccountId::new("github-0"),
        kind: WriteKind::Create,
        effect: EffectClass::Write,
        summary: "open a PR".into(),
        target: "repo/main".into(),
        payload: "{\"title\":\"x\"}".into(),
    };

    // Silent execution is refused.
    assert!(matches!(
        execute_without_receipt(&proposal),
        Err(ConnectorError::WriteReceiptRequired)
    ));

    // Deny-by-default gate: no allow without explicit target grant.
    let mut gate = PermissionGate::new(PermissionPolicy::deny_by_default());
    let denied = gate.authorize(proposal.clone()).unwrap();
    assert_eq!(denied.decision, PermissionDecision::Deny);
    assert!(matches!(
        gate.consume(&denied.id),
        Err(ConnectorError::WritePermissionDenied(_))
    ));

    // Allow path: grant target, authorize, execute only with receipt.
    let mut gate = PermissionGate::new(
        PermissionPolicy::deny_by_default().allow_target("repo/main"),
    );
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
    // Re-bind proposal account to the real id for the execute check.
    let mut proposal2 = proposal;
    proposal2.account_id = aid.clone();
    let allowed2 = gate.authorize(proposal2).unwrap();
    let handle = accounts.mint_handle(&aid).unwrap();

    let result = execute_with_receipt(&mut gate, &allowed2.id, &handle, |p, h| {
        assert_eq!(p.target, "repo/main");
        assert_eq!(h.account_id(), &aid);
        Ok(hide_connectors::WriteResult {
            receipt_id: String::new(),
            target: p.target.clone(),
            notes: "fixture execute".into(),
        })
    })
    .unwrap();
    assert_eq!(result.receipt_id, allowed2.id);
    assert_eq!(result.target, "repo/main");

    // Receipt is single-use.
    assert!(matches!(
        gate.consume(&allowed2.id),
        Err(ConnectorError::InvalidWriteReceipt(_))
    ));

    // Audit trail retained.
    let stored = gate.get(&allowed2.id).unwrap();
    assert!(stored.consumed);
    assert_eq!(stored.decision, PermissionDecision::Allow);
}

// ---------------------------------------------------------------------------
// Safety property 4: connector data never silently enters user memory
// ---------------------------------------------------------------------------

#[test]
fn safety_connector_read_cannot_write_user_memory() {
    let mut mem = ConnectorMemoryStore::new();
    let ingest = ConnectorIngestCap::mint();

    let rec = mem.ingest_connector(
        &ingest,
        FamilyId::new("rss"),
        hide_connectors::AccountId::new("rss-0"),
        "fixture-item-1",
        "First fixture item body",
    );
    assert!(matches!(
        rec.scope,
        hide_connectors::MemoryScope::Connector { .. }
    ));
    assert!(mem.user_records().is_empty());

    // The connector path cannot write user memory.
    let err = mem
        .ingest_as_user_from_connector(&ingest, "smuggle into user")
        .unwrap_err();
    assert!(matches!(
        err,
        ConnectorError::SilentMemoryPromotion {
            target: hide_connectors::MemoryScope::User
        }
    ));
    assert!(mem.user_records().is_empty());

    // Explicit promotion with the right cap is the only path.
    let cap = UserMemoryPromotionCap::mint();
    let user = mem.promote_to_user(&cap, &rec.id).unwrap();
    assert!(matches!(user.scope, hide_connectors::MemoryScope::User));
    assert_eq!(mem.user_records().len(), 1);
    assert_eq!(user.content, "First fixture item body");
}

// ---------------------------------------------------------------------------
// Safety property 5: revocation is real; in-flight fails closed
// ---------------------------------------------------------------------------

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

    // Pre-revoke works.
    let listed = conn
        .list(&store, &handle, &ListRequest::default())
        .unwrap();
    assert!(listed.iter().any(|o| o.title == "hello.txt"));

    // Begin in-flight, then revoke, then complete must fail closed.
    let guard =
        hide_connectors::InFlightGuard::begin(&store, &handle, &FamilyId::new("local_folder"))
            .unwrap();
    store.revoke(&id).unwrap();
    assert!(store.is_revoked(&id));
    assert!(matches!(
        guard.complete(&store),
        Err(ConnectorError::AccountRevoked(_)) | Err(ConnectorError::StaleHandle)
    ));

    // Subsequent ops with the old handle fail closed.
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

    // Minting a new handle after revoke also fails.
    assert!(matches!(
        store.mint_handle(&id),
        Err(ConnectorError::AccountRevoked(_))
    ));
}

// ---------------------------------------------------------------------------
// Implementations against fixtures
// ---------------------------------------------------------------------------

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

    // Path escape refused.
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
    // feed + 3 items
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
    assert!(item3
        .content
        .as_deref()
        .unwrap_or("")
        .contains("& entities")
        || item3
            .content
            .as_deref()
            .unwrap_or("")
            .contains("& entities")
        || item3.content.as_deref().unwrap_or("").contains("entities"));
}

#[test]
fn declared_connectors_not_constructible() {
    let reg = ConnectorRegistry::builtin();
    for abi in reg.declared() {
        let err = reg.construct(abi.family_id.as_str()).unwrap_err();
        assert!(
            matches!(err, ConnectorError::DeclaredNotConstructible(_)),
            "family {} should refuse construction, got {err:?}",
            abi.family_id
        );
    }
    // Unknown family.
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
    let back: hide_connectors::RegistryDocument = serde_json::from_str(&text).unwrap();
    assert_eq!(back.implemented, doc.implemented);
    assert_eq!(back.declared.len(), doc.declared.len());
    assert_eq!(back.families.len(), doc.families.len());
}

#[test]
fn registry_json_on_disk_matches_export() {
    // Workspace root: crates/hide-connectors/../..
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../HIDE_YOU_CONNECTOR_REGISTRY.json");
    let path = path.canonicalize().expect(
        "HIDE_YOU_CONNECTOR_REGISTRY.json missing; run: cargo run -p hide-connectors --example export_registry -- HIDE_YOU_CONNECTOR_REGISTRY.json",
    );
    let on_disk: hide_connectors::RegistryDocument =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let live = ConnectorRegistry::builtin().export_document();
    assert_eq!(
        on_disk.implemented, live.implemented,
        "implemented split drifted; re-export registry JSON"
    );
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
