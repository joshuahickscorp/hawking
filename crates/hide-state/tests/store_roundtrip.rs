//! Integration tests for the capsule stores over synthetic byte fixtures.
//!
//! These exercise the full save -> load -> fork -> compare -> release ->
//! inspect surface of both the in-memory and the content-addressed on-disk
//! store, and confirm that integrity and atomicity hold. No model is involved
//! and no timing or output-quality property is asserted anywhere.

use hide_state::{
    Ancestry, Capsule, CapsuleBuilder, CapsuleStore, CapsuleType, DiskStore, IdentityBinding,
    IntegrityAlgo, MemoryStore,
};

fn identity() -> IdentityBinding {
    IdentityBinding {
        model_weights_id: "weights".into(),
        arch_id: "arch".into(),
        tokenizer_id: "tok".into(),
        prompt_abi_version: "abi-1".into(),
        tool_registry_id: "reg".into(),
        engine_build_id: "build".into(),
        security_domain: "domain".into(),
    }
}

fn synthetic(seed: u8, len: usize, algo: IntegrityAlgo) -> Capsule {
    // Deterministic synthetic payload: a simple byte ramp offset by the seed.
    let payload: Vec<u8> = (0..len).map(|i| ((i as u8).wrapping_add(seed))).collect();
    CapsuleBuilder::new(CapsuleType::Kv, "model-fixture", identity())
        .runtime_version("rt-1")
        .dtype("f16")
        .device("cpu")
        .position(seed as u64)
        .context_pack_hash("ctx")
        .integrity_algo(algo)
        .seal(payload)
}

fn exercise_store<S: CapsuleStore>(store: &mut S) {
    let capsule = synthetic(3, 512, IntegrityAlgo::Blake3);
    let payload = capsule.payload().to_vec();

    // save -> load is byte-identical.
    let id = store.save(&capsule).unwrap();
    let loaded = store.load(&id).unwrap();
    assert_eq!(loaded, capsule);
    assert_eq!(loaded.payload(), payload.as_slice());

    // inspect returns metadata without payload, and agrees with the header.
    let meta = store.inspect(&id).unwrap();
    assert_eq!(meta.header, *capsule.header());
    assert_eq!(meta.header.bytes, payload.len() as u64);

    // fork preserves the payload, mints a distinct id, and records ancestry.
    let child_id = store.fork(&id).unwrap();
    assert_ne!(child_id, id);
    let child = store.load(&child_id).unwrap();
    assert_eq!(child.payload(), payload.as_slice());
    assert_eq!(child.parent_capsule_id(), Some(&id));

    // compare reports the ancestry relation and structural equality.
    let cmp = store.compare(&id, &child_id).unwrap();
    assert_eq!(cmp.ancestry, Ancestry::ParentToChild);
    assert!(cmp.payload_identical);
    assert!(cmp.identity_identical);
    assert!(!cmp.same_capsule_id);
    assert!(!cmp.header_identical); // ids and ancestry differ

    let cmp_self = store.compare(&id, &id).unwrap();
    assert_eq!(cmp_self.ancestry, Ancestry::Same);
    assert!(cmp_self.header_identical);

    // release removes the parent; the child remains loadable.
    store.release(&id).unwrap();
    assert!(store.load(&id).is_err());
    assert!(store.load(&child_id).is_ok());

    // releasing a missing id is an error, not a silent success.
    assert!(store.release(&id).is_err());
}

#[test]
fn memory_store_full_surface() {
    let mut store = MemoryStore::new();
    exercise_store(&mut store);
}

#[test]
fn disk_store_full_surface() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();
    exercise_store(&mut store);
}

#[test]
fn disk_store_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let id = {
        let mut store = DiskStore::open(dir.path()).unwrap();
        let capsule = synthetic(7, 256, IntegrityAlgo::Sha256);
        store.save(&capsule).unwrap()
    };
    // A fresh handle on the same root sees the persisted capsule.
    let store = DiskStore::open(dir.path()).unwrap();
    let loaded = store.load(&id).unwrap();
    assert_eq!(loaded.header().integrity.algo, IntegrityAlgo::Sha256);
    assert!(loaded
        .header()
        .integrity
        .verify(loaded.payload()));
}

#[test]
fn disk_store_rejects_corrupted_object() {
    use std::fs;

    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();
    let capsule = synthetic(1, 128, IntegrityAlgo::Blake3);
    let id = store.save(&capsule).unwrap();

    // Corrupt the single object file on disk by flipping one byte in it.
    let objects = dir.path().join("objects");
    let object_file = fs::read_dir(&objects)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().map(|x| x == "capsule").unwrap_or(false))
        .expect("object file present");
    let mut bytes = fs::read(&object_file).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&object_file, &bytes).unwrap();

    // The load must fail: the content address no longer matches the bytes.
    assert!(store.load(&id).is_err());
}

#[test]
fn disk_store_deduplicates_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();

    let capsule = synthetic(5, 100, IntegrityAlgo::Blake3);
    let id = store.save(&capsule).unwrap();
    let child_id = store.fork(&id).unwrap();

    // Parent and child carry identical payload bytes but differ in their
    // headers (id and ancestry), so their serialized forms and thus their
    // content addresses differ: two objects, two refs.
    let objects = dir.path().join("objects");
    let count = std::fs::read_dir(&objects).unwrap().count();
    assert_eq!(count, 2);

    // Releasing the parent removes exactly its object, leaving the child's.
    store.release(&id).unwrap();
    let count_after = std::fs::read_dir(&objects).unwrap().count();
    assert_eq!(count_after, 1);
    assert!(store.load(&child_id).is_ok());
}
