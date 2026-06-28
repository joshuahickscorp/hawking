//! Encryption-at-rest and on-disk layout enforcement (bible ch.10 §4.4, §4.1,
//! S6/S12).
//!
//! Two real capabilities:
//!
//!   1. **AES-256-GCM AEAD at rest.** A random 256-bit *workspace data key*
//!      (WDK) is wrapped by a key held in the OS keychain (`keyring`, behind the
//!      `os-keychain` feature); without that feature a clearly-marked
//!      file-backed dev store stands in. Per-store subkeys are HKDF-style
//!      derived (`blake3` keyed-hash, domain-separated) from the WDK, and every
//!      segment gets a fresh random 96-bit nonce stored beside its ciphertext
//!      (§4.4). Open is authenticated: a tampered segment fails the GCM tag.
//!
//!   2. **Layout validation that fails CLOSED.** [`validate_layout`] enforces
//!      that `.hide` is `0700` (owner-only) and that `.hide/log` is
//!      append-only / not agent-writable (§4.1, §4.5.2). A violation is an
//!      error the host surfaces — it never silently downgrades to plaintext or
//!      an open log (S12).

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use hide_core::error::{HideError, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const WDK_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
const WRAP_AAD: &[u8] = b"hide.atrest.wdk.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtRestPolicy {
    pub enabled: bool,
    /// Opaque handle into the keychain (`~/.hawking/keys/atrest.wrapkey.ref`,
    /// §4.1) — never key material.
    pub key_ref: Option<String>,
    pub encrypt_event_log: bool,
    pub encrypt_blobs: bool,
    pub encrypt_metadata: bool,
    /// `cache/`/`tmp/` may stay plaintext for speed (§4.4 / §9 Q4).
    pub plaintext_cache_allowed: bool,
}

impl Default for AtRestPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            key_ref: None,
            encrypt_event_log: false,
            encrypt_blobs: false,
            encrypt_metadata: false,
            plaintext_cache_allowed: true,
        }
    }
}

impl AtRestPolicy {
    /// The fully-on posture: encrypt log/blobs/metadata, leave derivable caches
    /// plaintext.
    pub fn enabled(key_ref: impl Into<String>) -> Self {
        Self {
            enabled: true,
            key_ref: Some(key_ref.into()),
            encrypt_event_log: true,
            encrypt_blobs: true,
            encrypt_metadata: true,
            plaintext_cache_allowed: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Keychain wrap-key store (real where the feature is on; file-backed dev store
// otherwise — both behind one trait so the rest of the crate is agnostic).
// ---------------------------------------------------------------------------

/// Stores/loads the *wrap key* that protects the WDK. The wrap key never leaves
/// this boundary in the clear; the WDK is sealed under it via AES-256-GCM.
pub trait WrapKeyStore: Send + Sync {
    /// Fetch the 32-byte wrap key for `key_ref`, creating it if absent.
    fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]>;
    /// Remove a wrap key (key rotation / workspace deletion).
    fn delete(&self, key_ref: &str) -> Result<()>;
}

/// macOS Keychain-backed wrap-key store (Data Protection keychain via
/// `apple-native`). Compiled only with the `os-keychain` feature.
#[cfg(feature = "os-keychain")]
#[derive(Debug, Clone)]
pub struct KeychainWrapKeyStore {
    service: String,
}

#[cfg(feature = "os-keychain")]
impl KeychainWrapKeyStore {
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }
}

#[cfg(feature = "os-keychain")]
impl Default for KeychainWrapKeyStore {
    fn default() -> Self {
        Self::new("com.hawking.hide.atrest")
    }
}

#[cfg(feature = "os-keychain")]
impl WrapKeyStore for KeychainWrapKeyStore {
    fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]> {
        let entry = keyring::Entry::new(&self.service, key_ref)
            .map_err(|e| HideError::Storage(format!("keychain entry: {e}")))?;
        match entry.get_password() {
            Ok(hex) => {
                let raw = hex_decode(&hex)?;
                let arr: [u8; WDK_LEN] = raw
                    .try_into()
                    .map_err(|_| HideError::Storage("wrap key wrong length".into()))?;
                Ok(arr)
            }
            Err(keyring::Error::NoEntry) => {
                let mut key = [0u8; WDK_LEN];
                rand::thread_rng().fill_bytes(&mut key);
                entry
                    .set_password(&hex_encode(&key))
                    .map_err(|e| HideError::Storage(format!("keychain set: {e}")))?;
                Ok(key)
            }
            Err(e) => Err(HideError::Storage(format!("keychain get: {e}"))),
        }
    }

    fn delete(&self, key_ref: &str) -> Result<()> {
        let entry = keyring::Entry::new(&self.service, key_ref)
            .map_err(|e| HideError::Storage(format!("keychain entry: {e}")))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(HideError::Storage(format!("keychain delete: {e}"))),
        }
    }
}

/// File-backed wrap-key store for environments without the OS keychain (CI,
/// Linux dev). The wrap key lives in a `0600` file under `keys/`. This is
/// **NOT** production-grade key protection — it is a documented dev stand-in so
/// the AEAD path is exercisable without a Keychain; the `os-keychain` feature
/// swaps in the real device-bound store.
#[derive(Debug, Clone)]
pub struct FileWrapKeyStore {
    dir: PathBuf,
}

impl FileWrapKeyStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, key_ref: &str) -> PathBuf {
        // key_ref is a controlled handle, but sanitize defensively.
        let safe: String = key_ref
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        self.dir.join(format!("{safe}.wrapkey"))
    }
}

impl WrapKeyStore for FileWrapKeyStore {
    fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]> {
        let path = self.path(key_ref);
        if path.exists() {
            let raw = std::fs::read(&path)?;
            let arr: [u8; WDK_LEN] = raw
                .try_into()
                .map_err(|_| HideError::Storage("wrap key wrong length".into()))?;
            return Ok(arr);
        }
        std::fs::create_dir_all(&self.dir)?;
        let mut key = [0u8; WDK_LEN];
        rand::thread_rng().fill_bytes(&mut key);
        std::fs::write(&path, key)?;
        set_owner_only_file(&path)?;
        Ok(key)
    }

    fn delete(&self, key_ref: &str) -> Result<()> {
        let path = self.path(key_ref);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The at-rest cipher: WDK sealed under the wrap key; per-store subkeys; per-
// segment nonces.
// ---------------------------------------------------------------------------

/// A sealed (wrapped) workspace data key, as persisted beside `key_ref`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrappedWdk {
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// A self-describing encrypted segment: nonce + AES-256-GCM ciphertext (incl.
/// the 16-byte auth tag). Stored beside the plaintext's slot (§4.4 / §4.2.2
/// `atrest_nonce`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptedSegment {
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// Holds an *open* WDK in memory and derives per-store subkeys. The WDK is zero
/// when this is dropped is best-effort (no `zeroize` dep here); the threat model
/// (§4.4) explicitly does not defend a running same-uid process — that is the
/// sandbox's job.
pub struct AtRestCipher {
    wdk: [u8; WDK_LEN],
}

impl std::fmt::Debug for AtRestCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtRestCipher").field("wdk", &"<sealed>").finish()
    }
}

impl AtRestCipher {
    /// Generate a fresh random WDK (on first enable).
    pub fn generate() -> Self {
        let mut wdk = [0u8; WDK_LEN];
        rand::thread_rng().fill_bytes(&mut wdk);
        Self { wdk }
    }

    pub fn from_wdk(wdk: [u8; WDK_LEN]) -> Self {
        Self { wdk }
    }

    /// Seal the WDK under the wrap key for at-rest persistence.
    pub fn wrap_wdk(&self, wrap_key: &[u8; WDK_LEN]) -> Result<WrappedWdk> {
        let nonce = random_nonce();
        let cipher = aes(wrap_key);
        let ct = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload { msg: &self.wdk, aad: WRAP_AAD },
            )
            .map_err(|_| HideError::Storage("WDK wrap failed".into()))?;
        Ok(WrappedWdk { nonce, ciphertext: ct })
    }

    /// Recover the WDK from its wrapped form (authenticated — a tampered wrap or
    /// the wrong wrap key fails).
    pub fn unwrap_wdk(wrap_key: &[u8; WDK_LEN], wrapped: &WrappedWdk) -> Result<Self> {
        let cipher = aes(wrap_key);
        let pt = cipher
            .decrypt(
                Nonce::from_slice(&wrapped.nonce),
                Payload { msg: &wrapped.ciphertext, aad: WRAP_AAD },
            )
            .map_err(|_| HideError::Storage("WDK unwrap failed (bad key or tampered)".into()))?;
        let wdk: [u8; WDK_LEN] = pt
            .try_into()
            .map_err(|_| HideError::Storage("unwrapped WDK wrong length".into()))?;
        Ok(Self { wdk })
    }

    /// HKDF-style per-store subkey: keyed-blake3 over the store id. Distinct
    /// stores (log/blobs/meta) never share a key, so a nonce reuse in one store
    /// can't endanger another (§4.4 `HKDF(WDK, context=store-id)`).
    fn subkey(&self, store_id: &str) -> [u8; WDK_LEN] {
        let mut h = blake3::Hasher::new_keyed(&self.wdk);
        h.update(b"hide.atrest.subkey.v1");
        h.update(store_id.as_bytes());
        *h.finalize().as_bytes()
    }

    /// Encrypt a segment of `store_id` with a fresh per-segment nonce. `store_id`
    /// is bound as AAD so a ciphertext can't be replayed into another store.
    pub fn encrypt_segment(&self, store_id: &str, plaintext: &[u8]) -> Result<EncryptedSegment> {
        let key = self.subkey(store_id);
        let cipher = aes(&key);
        let nonce = random_nonce();
        let ct = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload { msg: plaintext, aad: store_id.as_bytes() },
            )
            .map_err(|_| HideError::Storage("segment encrypt failed".into()))?;
        Ok(EncryptedSegment { nonce, ciphertext: ct })
    }

    /// Decrypt + authenticate a segment. A wrong `store_id`, wrong key, or any
    /// tamper fails the GCM tag (S12: an unauthenticated open is impossible).
    pub fn decrypt_segment(&self, store_id: &str, segment: &EncryptedSegment) -> Result<Vec<u8>> {
        let key = self.subkey(store_id);
        let cipher = aes(&key);
        cipher
            .decrypt(
                Nonce::from_slice(&segment.nonce),
                Payload { msg: &segment.ciphertext, aad: store_id.as_bytes() },
            )
            .map_err(|_| HideError::Storage("segment decrypt failed (bad key or tampered)".into()))
    }
}

fn aes(key: &[u8; WDK_LEN]) -> Aes256Gcm {
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
}

fn random_nonce() -> [u8; NONCE_LEN] {
    let mut n = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut n);
    n
}

// ---------------------------------------------------------------------------
// Layout validation (S12: fail CLOSED on 0700 / append-only violations).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutValidation {
    pub ok: bool,
    /// `.hide` is mode 0700 (no group/world bits).
    pub root_mode_owner_only: bool,
    /// `.hide/log` is NOT writable by group/other (agent-unreadable/append-only
    /// posture). `true` here means the *violation* condition — the field name is
    /// preserved from the scaffold for compatibility; see [`Self::ok`].
    pub hide_log_agent_writable: bool,
    pub warnings: Vec<String>,
}

/// Validate `.hide` layout permissions, failing CLOSED (§4.1, §4.5.2, S12).
///
/// On macOS/Unix this checks real file modes: `.hide` must be `0700`, and
/// `.hide/log` must not be group/world-writable. A violation returns `ok:false`
/// with the specifics in `warnings`; the host treats `!ok` as a refusal to
/// start (it does not downgrade to an open log). On non-Unix the mode bits are
/// unavailable, so it reports the inability rather than claiming success.
pub fn validate_layout(hide_dir: &Path) -> LayoutValidation {
    let mut warnings = Vec::new();

    if !hide_dir.exists() {
        return LayoutValidation {
            ok: false,
            root_mode_owner_only: false,
            hide_log_agent_writable: false,
            warnings: vec![format!("{} does not exist", hide_dir.display())],
        };
    }

    let log_dir = hide_dir.join("log");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let root_ok = match std::fs::metadata(hide_dir) {
            Ok(m) => {
                let mode = m.permissions().mode() & 0o777;
                if mode != 0o700 {
                    warnings.push(format!(
                        ".hide mode is {:#o}, expected 0700 (owner-only)",
                        mode
                    ));
                    false
                } else {
                    true
                }
            }
            Err(e) => {
                warnings.push(format!("cannot stat .hide: {e}"));
                false
            }
        };

        // log dir: must not be group/world writable. If absent we don't fail
        // (a fresh workspace) but we note it.
        let log_violation = if log_dir.exists() {
            match std::fs::metadata(&log_dir) {
                Ok(m) => {
                    let mode = m.permissions().mode() & 0o777;
                    if mode & 0o022 != 0 {
                        warnings.push(format!(
                            ".hide/log mode is {:#o}; group/world write bits set (audit log must be agent-unwritable)",
                            mode
                        ));
                        true
                    } else {
                        false
                    }
                }
                Err(e) => {
                    warnings.push(format!("cannot stat .hide/log: {e}"));
                    true
                }
            }
        } else {
            false
        };

        return LayoutValidation {
            ok: root_ok && !log_violation,
            root_mode_owner_only: root_ok,
            hide_log_agent_writable: log_violation,
            warnings,
        };
    }

    #[cfg(not(unix))]
    {
        let _ = log_dir;
        warnings.push("layout mode checks are only enforced on Unix; refusing to claim 0700".into());
        LayoutValidation {
            ok: false,
            root_mode_owner_only: false,
            hide_log_agent_writable: false,
            warnings,
        }
    }
}

/// Create `.hide` with mode 0700 if missing, then validate. The host's first-run
/// path (§4.1 step 1). Returns the validation; `!ok` means refuse-to-start.
pub fn ensure_and_validate_layout(hide_dir: &Path) -> Result<LayoutValidation> {
    if !hide_dir.exists() {
        std::fs::create_dir_all(hide_dir)?;
        set_owner_only_dir(hide_dir)?;
    }
    Ok(validate_layout(hide_dir))
}

#[cfg(unix)]
fn set_owner_only_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_owner_only_dir(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_owner_only_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_owner_only_file(_path: &Path) -> Result<()> {
    Ok(())
}

// Hex helpers for keychain string round-trips (keychain stores text, not bytes).
#[cfg(feature = "os-keychain")]
fn hex_encode(bytes: &[u8]) -> String {
    crate::audit::hex_lower(bytes)
}

#[cfg(feature = "os-keychain")]
fn hex_decode(input: &str) -> Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        return Err(HideError::Storage("odd-length hex wrap key".into()));
    }
    let val = |b: u8| -> Result<u8> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(HideError::Storage("invalid hex wrap key".into())),
        }
    };
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(input.len() / 2);
    for pair in bytes.chunks_exact(2) {
        out.push((val(pair[0])? << 4) | val(pair[1])?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aead_round_trips_and_authenticates() {
        let cipher = AtRestCipher::generate();
        let seg = cipher.encrypt_segment("log", b"hello secret world").unwrap();
        let pt = cipher.decrypt_segment("log", &seg).unwrap();
        assert_eq!(pt, b"hello secret world");

        // Tamper the ciphertext → tag fails.
        let mut bad = seg.clone();
        bad.ciphertext[0] ^= 0xff;
        assert!(cipher.decrypt_segment("log", &bad).is_err());

        // Wrong store id (AAD) → fails even with the same WDK.
        assert!(cipher.decrypt_segment("blobs", &seg).is_err());
    }

    #[test]
    fn nonces_are_per_segment_unique() {
        let cipher = AtRestCipher::generate();
        let a = cipher.encrypt_segment("log", b"same").unwrap();
        let b = cipher.encrypt_segment("log", b"same").unwrap();
        assert_ne!(a.nonce, b.nonce, "each segment gets a fresh nonce");
        assert_ne!(a.ciphertext, b.ciphertext);
    }

    #[test]
    fn distinct_stores_use_distinct_keys() {
        let cipher = AtRestCipher::generate();
        // Encrypt under "log", attempt decrypt as "log" with same nonce bytes
        // but the AAD/subkey separation means a blobs-keyed open of a log
        // segment must fail (covered above). Here assert subkeys differ.
        let log = cipher.subkey("log");
        let blobs = cipher.subkey("blobs");
        assert_ne!(log, blobs);
    }

    #[test]
    fn wrap_unwrap_wdk_round_trip() {
        let wdk_cipher = AtRestCipher::generate();
        let wrap_key = [42u8; WDK_LEN];
        let wrapped = wdk_cipher.wrap_wdk(&wrap_key).unwrap();

        let recovered = AtRestCipher::unwrap_wdk(&wrap_key, &wrapped).unwrap();
        // Recovered WDK must produce identical subkeys.
        assert_eq!(recovered.subkey("log"), wdk_cipher.subkey("log"));

        // Wrong wrap key fails.
        assert!(AtRestCipher::unwrap_wdk(&[0u8; WDK_LEN], &wrapped).is_err());
    }

    #[test]
    fn file_wrap_key_store_persists() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileWrapKeyStore::new(dir.path().join("keys"));
        let k1 = store.get_or_create("atrest").unwrap();
        let k2 = store.get_or_create("atrest").unwrap();
        assert_eq!(k1, k2, "stable across calls");
        store.delete("atrest").unwrap();
        let k3 = store.get_or_create("atrest").unwrap();
        assert_ne!(k1, k3, "regenerated after delete");
    }

    #[test]
    fn full_wrap_key_to_segment_flow() {
        // End-to-end: wrap-key store → wrap WDK → persist → reopen → decrypt.
        let dir = tempfile::tempdir().unwrap();
        let store = FileWrapKeyStore::new(dir.path().join("keys"));
        let wrap_key = store.get_or_create("ws").unwrap();

        let cipher = AtRestCipher::generate();
        let wrapped = cipher.wrap_wdk(&wrap_key).unwrap();
        let seg = cipher.encrypt_segment("log", b"audit event bytes").unwrap();

        // Simulate reopen: reload wrap key, unwrap WDK, decrypt.
        let wrap_key2 = store.get_or_create("ws").unwrap();
        let cipher2 = AtRestCipher::unwrap_wdk(&wrap_key2, &wrapped).unwrap();
        let pt = cipher2.decrypt_segment("log", &seg).unwrap();
        assert_eq!(pt, b"audit event bytes");
    }

    #[cfg(unix)]
    #[test]
    fn layout_validation_enforces_0700() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let hide = dir.path().join(".hide");
        let v = ensure_and_validate_layout(&hide).unwrap();
        assert!(v.ok, "fresh 0700 .hide should validate: {:?}", v.warnings);
        assert!(v.root_mode_owner_only);

        // Loosen to 0755 → fail closed.
        let mut perms = std::fs::metadata(&hide).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&hide, perms).unwrap();
        let v2 = validate_layout(&hide);
        assert!(!v2.ok);
        assert!(!v2.root_mode_owner_only);
        assert!(v2.warnings.iter().any(|w| w.contains("0700")));
    }

    #[cfg(unix)]
    #[test]
    fn layout_validation_flags_writable_log() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let hide = dir.path().join(".hide");
        ensure_and_validate_layout(&hide).unwrap();
        let log = hide.join("log");
        std::fs::create_dir_all(&log).unwrap();
        // group/world-writable log dir.
        std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o777)).unwrap();
        let v = validate_layout(&hide);
        assert!(!v.ok);
        assert!(v.hide_log_agent_writable);
    }

    #[test]
    fn missing_hide_dir_fails_closed() {
        let v = validate_layout(Path::new("/nonexistent/.hide/xyz"));
        assert!(!v.ok);
    }
}
