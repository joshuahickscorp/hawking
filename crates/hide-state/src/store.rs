//! Capsule stores: a trait plus an in-memory impl and a content-addressed
//! on-disk impl.
//!
//! Both impls serialize through [`Capsule::to_bytes`] and parse back through
//! [`Capsule::from_bytes`], so every load is integrity-checked. The on-disk
//! store is content-addressed: an object is named by the digest of its bytes,
//! written atomically (temp file plus rename), and a small reference file maps
//! a capsule id to its content address. A load recomputes the content address
//! and rejects a mismatch before it even checks the payload digest.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use ulid::Ulid;

use crate::capsule::{Capsule, CapsuleInspect};
use crate::error::{CapsuleError, Result};
use crate::header::CapsuleId;

/// How two capsules relate through their recorded ancestry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ancestry {
    /// Both refer to the same capsule id.
    Same,
    /// The first is the parent of the second.
    ParentToChild,
    /// The second is the parent of the first.
    ChildToParent,
    /// Both were forked from the same parent.
    Siblings,
    /// No recorded relationship.
    Unrelated,
}

/// The result of comparing two capsules. Deliberately structural: it reports
/// what is and is not equal and how the two relate by ancestry, and makes no
/// judgement about which is preferable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleComparison {
    pub same_capsule_id: bool,
    pub payload_identical: bool,
    pub identity_identical: bool,
    pub header_identical: bool,
    pub ancestry: Ancestry,
}

impl CapsuleComparison {
    /// Compare two capsules field by field.
    pub fn of(a: &Capsule, b: &Capsule) -> CapsuleComparison {
        let a_id = a.capsule_id();
        let b_id = b.capsule_id();
        let a_parent = a.parent_capsule_id();
        let b_parent = b.parent_capsule_id();

        let ancestry = if a_id == b_id {
            Ancestry::Same
        } else if b_parent == Some(a_id) {
            Ancestry::ParentToChild
        } else if a_parent == Some(b_id) {
            Ancestry::ChildToParent
        } else if a_parent.is_some() && a_parent == b_parent {
            Ancestry::Siblings
        } else {
            Ancestry::Unrelated
        };

        CapsuleComparison {
            same_capsule_id: a_id == b_id,
            payload_identical: a.payload() == b.payload(),
            identity_identical: a.identity() == b.identity(),
            header_identical: a.header() == b.header(),
            ancestry,
        }
    }
}

/// A place to save, load, fork, compare, release, and inspect capsules.
///
/// `fork` and `compare` have default implementations in terms of `load` and
/// `save`, so an impl only has to provide the four primitive operations.
pub trait CapsuleStore {
    /// Save a capsule, keyed by its own id, and return that id. Saving a
    /// capsule whose id already exists overwrites the stored bytes.
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId>;

    /// Load and integrity-check the capsule with `id`.
    fn load(&self, id: &CapsuleId) -> Result<Capsule>;

    /// Release the capsule with `id`. Returns `NotFound` if it is absent.
    fn release(&mut self, id: &CapsuleId) -> Result<()>;

    /// Inspect the metadata of the capsule with `id` without materializing its
    /// payload.
    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect>;

    /// Fork the capsule with `id` and save the fork, returning the new id.
    fn fork(&mut self, id: &CapsuleId) -> Result<CapsuleId> {
        let source = self.load(id)?;
        let forked = source.fork();
        self.save(&forked)
    }

    /// Compare the two stored capsules with the given ids.
    fn compare(&self, a: &CapsuleId, b: &CapsuleId) -> Result<CapsuleComparison> {
        let ca = self.load(a)?;
        let cb = self.load(b)?;
        Ok(CapsuleComparison::of(&ca, &cb))
    }
}

/// An in-memory store. Holds the serialized bytes of each capsule, so a load
/// runs the same integrity checks a persistent store would.
#[derive(Debug, Default)]
pub struct MemoryStore {
    objects: HashMap<String, Vec<u8>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        MemoryStore {
            objects: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl CapsuleStore for MemoryStore {
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId> {
        let id = capsule.capsule_id().clone();
        self.objects.insert(id.0.clone(), capsule.to_bytes());
        Ok(id)
    }

    fn load(&self, id: &CapsuleId) -> Result<Capsule> {
        let bytes = self
            .objects
            .get(&id.0)
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))?;
        Capsule::from_bytes(bytes)
    }

    fn release(&mut self, id: &CapsuleId) -> Result<()> {
        self.objects
            .remove(&id.0)
            .map(|_| ())
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))
    }

    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect> {
        let bytes = self
            .objects
            .get(&id.0)
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))?;
        Capsule::inspect_bytes(bytes)
    }
}

/// A content-addressed on-disk store.
///
/// Layout under `root`:
///
/// - `objects/<content-address>.capsule` holds the serialized bytes, named by
///   the blake3 digest of those bytes.
/// - `refs/<capsule-id>` holds the content address the id currently points at.
///
/// Both files are written atomically by writing a uniquely named temp file in
/// the same directory and renaming it into place. Distinct capsules with
/// identical bytes share one object; releasing an id removes its ref and, if no
/// other ref points at that object, the object too.
#[derive(Debug, Clone)]
pub struct DiskStore {
    root: PathBuf,
}

impl DiskStore {
    /// Open (creating if needed) a store rooted at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("refs"))?;
        Ok(DiskStore { root })
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn refs_dir(&self) -> PathBuf {
        self.root.join("refs")
    }

    fn object_path(&self, address: &str) -> PathBuf {
        self.objects_dir().join(format!("{address}.capsule"))
    }

    fn ref_path(&self, id: &CapsuleId) -> PathBuf {
        self.refs_dir().join(&id.0)
    }

    fn content_address(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    /// Read a capsule id's content address from its ref file.
    fn read_ref(&self, id: &CapsuleId) -> Result<String> {
        match fs::read_to_string(self.ref_path(id)) {
            Ok(s) => Ok(s.trim().to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(CapsuleError::NotFound(id.0.clone()))
            }
            Err(e) => Err(CapsuleError::Io(e)),
        }
    }

    /// Read and content-verify the object at `address`.
    fn read_object(&self, address: &str) -> Result<Vec<u8>> {
        let path = self.object_path(address);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(CapsuleError::Corrupt {
                    detail: format!("object {address} referenced but missing"),
                });
            }
            Err(e) => return Err(CapsuleError::Io(e)),
        };
        let actual = Self::content_address(&bytes);
        if actual != address {
            return Err(CapsuleError::ContentAddressMismatch {
                expected: address.to_string(),
                actual,
            });
        }
        Ok(bytes)
    }
}

/// Write `bytes` to `path` atomically: write a uniquely named temp file in the
/// same directory, flush it, then rename it over `path`.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| CapsuleError::Corrupt {
            detail: "target path has no parent directory".to_string(),
        })?;
    let tmp = dir.join(format!(".tmp-{}", Ulid::new()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
    }
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best-effort cleanup of the temp file; report the rename error.
            let _ = fs::remove_file(&tmp);
            Err(CapsuleError::Io(e))
        }
    }
}

impl CapsuleStore for DiskStore {
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId> {
        let id = capsule.capsule_id().clone();
        let bytes = capsule.to_bytes();
        let address = Self::content_address(&bytes);
        atomic_write(&self.object_path(&address), &bytes)?;
        atomic_write(&self.ref_path(&id), address.as_bytes())?;
        Ok(id)
    }

    fn load(&self, id: &CapsuleId) -> Result<Capsule> {
        let address = self.read_ref(id)?;
        let bytes = self.read_object(&address)?;
        Capsule::from_bytes(&bytes)
    }

    fn release(&mut self, id: &CapsuleId) -> Result<()> {
        let address = self.read_ref(id)?;
        fs::remove_file(self.ref_path(id))?;
        // Garbage-collect the object if no other ref points at it.
        if !self.address_is_referenced(&address)? {
            let obj = self.object_path(&address);
            if obj.exists() {
                fs::remove_file(obj)?;
            }
        }
        Ok(())
    }

    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect> {
        let address = self.read_ref(id)?;
        let bytes = self.read_object(&address)?;
        Capsule::inspect_bytes(&bytes)
    }
}

impl DiskStore {
    /// Whether any ref file currently points at `address`.
    fn address_is_referenced(&self, address: &str) -> Result<bool> {
        for entry in fs::read_dir(self.refs_dir())? {
            let entry = entry?;
            if let Ok(contents) = fs::read_to_string(entry.path()) {
                if contents.trim() == address {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}
