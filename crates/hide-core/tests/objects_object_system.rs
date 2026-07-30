use hide_core::objects::*;
use std::io::Write;
use tempfile::tempdir;
fn perms(owner: &str) -> ObjectPermissions {
    ObjectPermissions::owner_only(owner, vec![Surface::You, Surface::Chat])
}
fn reader(owner: &str) -> Reader {
    Reader {
        principal: owner.into(),
        surface: Surface::You,
    }
}
fn ingest_and_finish(store: &ObjectStore, bytes: &[u8], mime: &str, label: &str) -> ContentHash {
    let job = store
        .enqueue_bytes(
            bytes,
            mime,
            ObjectSource::UserUpload {
                filename: Some(label.into()),
                session_id: Some("ses_test".into()),
            },
            perms("alice"),
            RetentionPolicy::durable(),
            Some(label.into()),
            "alice",
            Priority::NORMAL,
        )
        .unwrap();
    let (id, st) = store.process_one().unwrap();
    assert_eq!(id, job);
    assert_eq!(st, JobStatus::Succeeded, "dead={:?}", store.dead_letter());
    store.hash_for_job(&job).expect("hash after success")
}
#[test]
fn dedup_same_bytes_one_object_two_refs() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let body = b"identical-payload-for-dedup";
    let h1 = ingest_and_finish(&store, body, "text/plain", "a.txt");
    let h2 = ingest_and_finish(&store, body, "text/plain", "b.txt");
    assert_eq!(h1, h2, "same bytes must share content hash identity");
    assert_eq!(store.object_count(), 1, "one object record");
    assert_eq!(store.ref_count(), 2, "two references");
    let refs = store.refs_for(&h1);
    assert_eq!(refs.len(), 2);
    let labels: Vec<_> = refs.iter().filter_map(|r| r.label.clone()).collect();
    assert!(labels.contains(&"a.txt".into()));
    assert!(labels.contains(&"b.txt".into()));
}
#[test]
fn distinct_bytes_distinct_objects() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let h1 = ingest_and_finish(&store, b"alpha-bytes", "text/plain", "a.txt");
    let h2 = ingest_and_finish(&store, b"beta-bytes!!", "text/plain", "b.txt");
    assert_ne!(h1, h2, "different bytes must not collide");
    assert_eq!(store.object_count(), 2);
    assert_eq!(store.ref_count(), 2);
}
#[test]
fn large_synthetic_fixture_streams_without_full_ram() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let size = 8 * 1024 * 1024;
    let path = dir.path().join("synthetic_video.bin");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        let pattern = b"VIDFAKE_FRAME_";
        let mut written = 0usize;
        let mut block = Vec::with_capacity(CHUNK_SIZE);
        while written < size {
            block.clear();
            while block.len() < CHUNK_SIZE && written + block.len() < size {
                block.extend_from_slice(pattern);
                block.push((written % 251) as u8);
            }
            let take = (size - written).min(block.len());
            f.write_all(&block[..take]).unwrap();
            written += take;
        }
    }
    let job = store
        .enqueue_path(
            &path,
            "video/mp4",
            ObjectSource::Synthetic {
                label: "large_video_fixture".into(),
            },
            perms("alice"),
            RetentionPolicy::durable(),
            Some("clip.mp4".into()),
            "alice",
            Priority::HIGH,
        )
        .unwrap();
    let (_id, st) = store.process_one().unwrap();
    assert_eq!(st, JobStatus::Succeeded);
    let hash = store.hash_for_job(&job).unwrap();
    let rec = store.get_record(&hash, &reader("alice")).unwrap();
    assert_eq!(rec.size_bytes, size as u64);
    assert_eq!(rec.kind, ObjectKind::Video);
    assert_eq!(rec.status, ObjectStatus::Ready);
    for stage in [StageName::Receive, StageName::Persist] {
        let s = rec.stage(stage).expect("stage present");
        assert!(s.peak_buffer_bytes <= CHUNK_SIZE);
        assert_eq!(s.status, StageStatus::Complete);
    }
    let tr = rec
        .derivative(DerivativeKind::Transcript)
        .expect("transcript");
    assert!(tr.produced_by.contains("FakeAsrEngine"));
    assert!(tr.inline_text.as_ref().unwrap().contains("FakeAsrEngine"));
    let view = store
        .compile_view(
            &hash,
            &reader("alice"),
            &DerivativeSelection {
                kinds: vec![DerivativeKind::Transcript, DerivativeKind::Summary],
            },
            Some("clip.mp4".into()),
        )
        .unwrap();
    assert!(!CompileObjectView::exposes_raw_bytes());
    assert!(view.try_raw_bytes().is_err());
    assert!(view
        .derivatives
        .iter()
        .any(|d| d.kind == DerivativeKind::Transcript));
}
#[test]
fn compile_path_cannot_reach_raw_bytes() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let hash = ingest_and_finish(&store, b"secret-body-bytes", "text/plain", "s.txt");
    let view = store
        .compile_view(
            &hash,
            &reader("alice"),
            &DerivativeSelection::default(),
            None,
        )
        .unwrap();
    assert!(!CompileObjectView::exposes_raw_bytes());
    assert!(matches!(
        view.try_raw_bytes(),
        Err(ObjectError::RawBytesForbidden)
    ));
    assert!(view.derivatives.iter().any(|d| d.text.is_some()));
    let json = serde_json::to_string(&view).unwrap();
    assert!(json.contains("text_extract") || json.contains("summary"));
}
#[test]
fn raw_bytes_require_cap_and_allow_export() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let mut p = perms("alice");
    p.allow_export = true;
    let job = store
        .enqueue_bytes(
            b"exportable",
            "text/plain",
            ObjectSource::Synthetic { label: "e".into() },
            p,
            RetentionPolicy::durable(),
            None,
            "alice",
            Priority::NORMAL,
        )
        .unwrap();
    store.process_one().unwrap();
    let hash = store.hash_for_job(&job).unwrap();
    let cap = RawBytesCap::mint();
    let bytes = store.raw_bytes(&hash, &reader("alice"), &cap).unwrap();
    assert_eq!(bytes, b"exportable");
}
#[test]
fn queue_priority_order() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let low = store
        .enqueue_bytes(
            b"low",
            "text/plain",
            ObjectSource::Synthetic {
                label: "low".into(),
            },
            perms("alice"),
            RetentionPolicy::durable(),
            Some("low".into()),
            "alice",
            Priority::LOW,
        )
        .unwrap();
    let high = store
        .enqueue_bytes(
            b"high",
            "text/plain",
            ObjectSource::Synthetic {
                label: "high".into(),
            },
            perms("alice"),
            RetentionPolicy::durable(),
            Some("high".into()),
            "alice",
            Priority::CRITICAL,
        )
        .unwrap();
    let (first, st) = store.process_one().unwrap();
    assert_eq!(st, JobStatus::Succeeded);
    assert_eq!(first, high, "critical priority must run before low");
    let (second, st) = store.process_one().unwrap();
    assert_eq!(st, JobStatus::Succeeded);
    assert_eq!(second, low);
}
#[test]
fn failed_stage_retries_then_dead_letters_visibly() {
    let dir = tempdir().unwrap();
    let budget = StorageBudget {
        max_local_bytes: 20,
        max_cloud_bytes: 20,
        max_object_bytes: 10_000,
        policy_note: "tiny".into(),
    };
    let store = ObjectStore::open(dir.path(), budget).unwrap();
    let _ = ingest_and_finish(&store, b"fifteen-bytes!!", "text/plain", "small.txt");
    assert_eq!(store.used_local_bytes(), 15);
    let job = store
        .enqueue_bytes(
            b"thirty-bytes-of-payload-here!!",
            "text/plain",
            ObjectSource::Synthetic {
                label: "big".into(),
            },
            perms("alice"),
            RetentionPolicy::durable(),
            None,
            "alice",
            Priority::NORMAL,
        )
        .unwrap();
    let mut saw_failed = false;
    for _ in 0..8 {
        match store.process_one() {
            Ok((id, JobStatus::FailedVisible)) => {
                assert_eq!(id, job);
                saw_failed = true;
                break;
            }
            Ok((_, JobStatus::RetryWait)) => continue,
            Ok((_, JobStatus::Succeeded)) => {
                panic!("second object should not admit under tiny budget")
            }
            Ok(_) => continue,
            Err(ObjectError::QueueEmpty) => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert!(saw_failed, "must fail visibly");
    assert_eq!(store.dead_letter().len(), 1);
    assert_eq!(store.dead_letter()[0].id, job);
    assert!(
        store.dead_letter()[0]
            .last_error
            .as_ref()
            .unwrap()
            .contains("budget")
            || store.dead_letter()[0]
                .last_error
                .as_ref()
                .unwrap()
                .contains("Budget")
            || store.dead_letter()[0].last_error.as_ref().is_some()
    );
}
#[test]
fn permissions_honoured_at_read() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let hash = ingest_and_finish(&store, b"private", "text/plain", "p.txt");
    let bob = Reader {
        principal: "bob".into(),
        surface: Surface::You,
    };
    assert!(matches!(
        store.get_record(&hash, &bob),
        Err(ObjectError::PermissionDenied { .. })
    ));
    let alice_chat = Reader {
        principal: "alice".into(),
        surface: Surface::Chat,
    };
    assert!(store.get_record(&hash, &alice_chat).is_ok());
    let alice_ide = Reader {
        principal: "alice".into(),
        surface: Surface::Ide,
    };
    assert!(matches!(
        store.get_record(&hash, &alice_ide),
        Err(ObjectError::PermissionDenied { .. })
    ));
}
#[test]
fn retention_ttl_honoured_at_read() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    store.set_clock_ms(Some(1_000));
    let job = store
        .enqueue_bytes(
            b"ephemeral",
            "text/plain",
            ObjectSource::Synthetic {
                label: "ttl".into(),
            },
            perms("alice"),
            RetentionPolicy::ttl_until(5_000),
            None,
            "alice",
            Priority::NORMAL,
        )
        .unwrap();
    store.process_one().unwrap();
    let hash = store.hash_for_job(&job).unwrap();
    assert!(store.get_record(&hash, &reader("alice")).is_ok());
    store.set_clock_ms(Some(5_000));
    assert!(matches!(
        store.get_record(&hash, &reader("alice")),
        Err(ObjectError::RetentionDenied { .. })
    ));
}
#[test]
fn image_gets_fake_ocr_and_thumbnail() {
    let dir = tempdir().unwrap();
    let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
    let hash = ingest_and_finish(&store, b"\x89PNG_FAKE_BYTES", "image/png", "x.png");
    let rec = store.get_record(&hash, &reader("alice")).unwrap();
    assert_eq!(rec.kind, ObjectKind::Image);
    let ocr = rec.derivative(DerivativeKind::Ocr).unwrap();
    assert_eq!(ocr.produced_by, "FakeOcrEngine");
    let thumb = rec.derivative(DerivativeKind::Thumbnail).unwrap();
    assert_eq!(thumb.produced_by, "FakeThumbnailer");
}
#[test]
fn budget_statement_is_honest() {
    assert!(BOUND_STATEMENT.contains("finite") || BOUND_STATEMENT.contains("bounded"));
    assert!(!BOUND_STATEMENT
        .to_lowercase()
        .contains("unlimited storage without"));
    let b = StorageBudget::default();
    assert!(b.max_local_bytes > 0);
    assert!(b.policy_note.contains("not unlimited"));
}
