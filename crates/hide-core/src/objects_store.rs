//! Content-addressed object store.
//!
//! Identity is [`ContentHash`]. The same bytes ingested twice yield one
//! [`ObjectRecord`] and two [`ObjectRef`]s.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::objects::budget::StorageBudget;
use crate::objects::derivatives::{CompileObjectView, DerivativeSelection, RawBytesCap};
use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::{ContentHash, CHUNK_SIZE};
use crate::objects::permissions::{ObjectPermissions, Reader};
use crate::objects::pipeline::{self, new_processing_record};
use crate::objects::processors::ProcessorSet;
use crate::objects::queue::{IngestJob, IngestQueue, JobStatus, Priority};
use crate::objects::retention::RetentionPolicy;
use crate::objects::schema::{
    ObjectLocation, ObjectRecord, ObjectRef, ObjectSource, ObjectStatus, RefId, StageName,
};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

struct Inner {
    root: PathBuf,
    budget: StorageBudget,
    /// content_hash string → record
    objects: BTreeMap<String, ObjectRecord>,
    /// ref_id → ObjectRef
    refs: BTreeMap<String, ObjectRef>,
    /// job_id → staging path
    staging: BTreeMap<String, PathBuf>,
    /// job_id → in-progress ObjectRecord (before hash known / after)
    in_flight: BTreeMap<String, ObjectRecord>,
    used_local_bytes: u64,
    queue: IngestQueue,
    live_session: Option<String>,
    clock_ms: Option<u64>,
}

/// Durable, restart-safe portion of an [`ObjectStore`].
///
/// The queue and staging files deliberately remain ephemeral: callers can
/// retry an interrupted ingestion explicitly and every completed object/ref is
/// persisted atomically before `process_one` reports success.  Persisting only
/// ready records also avoids reopening a half-written staging file as if it
/// were model-readable evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    schema: String,
    #[serde(default)]
    objects: BTreeMap<String, ObjectRecord>,
    #[serde(default)]
    refs: BTreeMap<String, ObjectRef>,
    #[serde(default)]
    used_local_bytes: u64,
}

impl PersistedState {
    const SCHEMA: &'static str = "hide.objects.state.v1";

    fn empty() -> Self {
        Self {
            schema: Self::SCHEMA.to_string(),
            objects: BTreeMap::new(),
            refs: BTreeMap::new(),
            used_local_bytes: 0,
        }
    }
}

impl Inner {
    fn now(&self) -> u64 {
        self.clock_ms.unwrap_or_else(now_ms)
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let name = hash.as_str().trim_start_matches("blake3:");
        self.objects_dir().join(name)
    }

    fn staging_path(&self, job_id: &str) -> PathBuf {
        self.root.join("staging").join(job_id)
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("meta").join("state.json")
    }
}

/// The YOU object store: queue, content-addressed blobs, pipeline, compile view.
pub struct ObjectStore {
    inner: Arc<Mutex<Inner>>,
    processors: ProcessorSet,
}

impl ObjectStore {
    /// Open (or create) a store at `root` with the given budget.
    pub fn open(root: impl AsRef<Path>, budget: StorageBudget) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("derivatives"))?;
        fs::create_dir_all(root.join("staging"))?;
        fs::create_dir_all(root.join("meta"))?;
        let state_path = root.join("meta").join("state.json");
        let state = if state_path.exists() {
            let bytes = fs::read(&state_path)?;
            let state: PersistedState = serde_json::from_slice(&bytes).map_err(|error| {
                ObjectError::Invalid(format!(
                    "could not read persisted object metadata at {}: {error}",
                    state_path.display()
                ))
            })?;
            if state.schema != PersistedState::SCHEMA {
                return Err(ObjectError::Invalid(format!(
                    "unsupported object metadata schema {:?} at {}",
                    state.schema,
                    state_path.display()
                )));
            }
            state
        } else {
            PersistedState::empty()
        };

        // Do not trust a stored accounting number over the durable records.
        // A metadata-only edit cannot make the next admission overcommit the
        // configured local-storage budget.
        let used_local_bytes = state.objects.values().map(|record| record.size_bytes).sum();

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                root,
                budget,
                objects: state.objects,
                refs: state.refs,
                staging: BTreeMap::new(),
                in_flight: BTreeMap::new(),
                used_local_bytes,
                queue: IngestQueue::new(),
                live_session: None,
                clock_ms: None,
            })),
            processors: ProcessorSet::fake_defaults(),
        })
    }

    /// Persist ready object/ref metadata using a same-directory rename.  The
    /// body was already content-addressed by the Persist stage; this metadata
    /// commit is what makes it discoverable after a new HCLI process opens the
    /// workspace.  A failed commit is surfaced to the caller, never hidden.
    fn persist_state(inner: &Inner) -> Result<()> {
        let state = PersistedState {
            schema: PersistedState::SCHEMA.to_string(),
            objects: inner.objects.clone(),
            refs: inner.refs.clone(),
            used_local_bytes: inner.used_local_bytes,
        };
        let bytes = serde_json::to_vec_pretty(&state)
            .map_err(|error| ObjectError::Invalid(format!("serialize object metadata: {error}")))?;
        let path = inner.state_path();
        let tmp = path.with_extension(format!("{}.tmp", ulid::Ulid::new()));
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    pub fn set_clock_ms(&self, ms: Option<u64>) {
        self.inner.lock().clock_ms = ms;
    }

    pub fn set_live_session(&self, session_id: Option<String>) {
        self.inner.lock().live_session = session_id;
    }

    pub fn used_local_bytes(&self) -> u64 {
        self.inner.lock().used_local_bytes
    }

    pub fn budget(&self) -> StorageBudget {
        self.inner.lock().budget.clone()
    }

    pub fn queue_len(&self) -> usize {
        self.inner.lock().queue.len()
    }

    pub fn dead_letter(&self) -> Vec<IngestJob> {
        self.inner.lock().queue.dead_letter().to_vec()
    }

    // ------------------------------------------------------------------
    // Ingest API
    // ------------------------------------------------------------------

    /// Enqueue a byte body. Staging is written in [`CHUNK_SIZE`] slices so the
    /// caller can pass a large slice without the store keeping a second copy
    /// beyond the staging file.
    pub fn enqueue_bytes(
        &self,
        bytes: &[u8],
        mime: impl Into<String>,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: impl Into<String>,
        priority: Priority,
    ) -> Result<String> {
        let mime = mime.into();
        let created_by = created_by.into();
        let mut g = self.inner.lock();
        let now = g.now();
        g.budget.check_object_size(bytes.len() as u64)?;

        let job = IngestJob::new(
            priority,
            0,
            mime.clone(),
            source.clone(),
            permissions.clone(),
            retention.clone(),
            label,
            created_by,
            now,
        );
        let job_id = job.id.clone();
        let stage_path = g.staging_path(&job_id);
        write_bytes_chunked(&stage_path, bytes)?;
        g.staging.insert(job_id.clone(), stage_path);
        g.in_flight.insert(
            job_id.clone(),
            new_processing_record(mime, source, permissions, retention, now),
        );
        g.queue.enqueue(job);
        Ok(job_id)
    }

    /// Enqueue from an existing file path (streamed; not loaded whole into RAM
    /// by the store beyond chunked copy into staging).
    pub fn enqueue_path(
        &self,
        path: impl AsRef<Path>,
        mime: impl Into<String>,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: impl Into<String>,
        priority: Priority,
    ) -> Result<String> {
        let mime = mime.into();
        let created_by = created_by.into();
        let src = path.as_ref();
        let meta = fs::metadata(src)?;
        let mut g = self.inner.lock();
        let now = g.now();
        g.budget.check_object_size(meta.len())?;

        let job = IngestJob::new(
            priority,
            0,
            mime.clone(),
            source.clone(),
            permissions.clone(),
            retention.clone(),
            label,
            created_by,
            now,
        );
        let job_id = job.id.clone();
        let stage_path = g.staging_path(&job_id);
        copy_chunked(src, &stage_path)?;
        g.staging.insert(job_id.clone(), stage_path);
        g.in_flight.insert(
            job_id.clone(),
            new_processing_record(mime, source, permissions, retention, now),
        );
        g.queue.enqueue(job);
        Ok(job_id)
    }

    /// Process one job through **all** remaining stages (or until a stage fails).
    /// Returns the job id and final job status.
    pub fn process_one(&self) -> Result<(String, JobStatus)> {
        let mut job = {
            let mut g = self.inner.lock();
            g.queue.pop_ready()?
        };
        let job_id = job.id.clone();

        loop {
            let outcome = self.run_job_stage(&mut job);
            match outcome {
                Ok(done) if done => {
                    let mut g = self.inner.lock();
                    let now = g.now();
                    // Merge in-flight record into objects map (dedup by hash).
                    if let Some(rec) = g.in_flight.remove(&job_id) {
                        // Keep a copy until the metadata commit succeeds.  If
                        // that commit fails, the job goes through the normal
                        // visible retry/dead-letter path rather than being
                        // silently lost after the raw body was persisted.
                        let rec_for_retry = rec.clone();
                        let hash_key = rec.content_hash.as_str().to_string();
                        job.content_hash = Some(rec.content_hash.clone());
                        let is_new = !g.objects.contains_key(&hash_key);
                        if is_new {
                            // Account storage only for new bodies.
                            if let Err(e) = g
                                .budget
                                .check_local_admission(g.used_local_bytes, rec.size_bytes)
                            {
                                // Keep in_flight so retries can re-attempt admission.
                                g.in_flight.insert(job_id.clone(), rec);
                                let st = g.queue.fail_stage(
                                    job,
                                    StageName::Finalize,
                                    e.to_string(),
                                    now,
                                );
                                return Ok((job_id, st));
                            }
                            g.used_local_bytes = g.used_local_bytes.saturating_add(rec.size_bytes);
                            g.objects.insert(hash_key.clone(), rec);
                        }
                        // Always create a new ref (dedup = same object, extra ref).
                        let r = ObjectRef {
                            id: RefId::new(),
                            content_hash: ContentHash(hash_key.clone()),
                            label: job.label.clone(),
                            created_at_ms: now,
                            created_by: job.created_by.clone(),
                        };
                        let ref_id = r.id.as_str().to_string();
                        g.refs.insert(ref_id.clone(), r);
                        // The durable metadata commit happens before the job
                        // is reported succeeded, so a new process can list or
                        // compile this evidence immediately after success.
                        if let Err(error) = Self::persist_state(&g) {
                            g.refs.remove(&ref_id);
                            if is_new {
                                g.objects.remove(&hash_key);
                                g.used_local_bytes =
                                    g.used_local_bytes.saturating_sub(rec_for_retry.size_bytes);
                            }
                            g.in_flight.insert(job_id.clone(), rec_for_retry);
                            let st = g.queue.fail_stage(
                                job,
                                StageName::Finalize,
                                error.to_string(),
                                now,
                            );
                            return Ok((job_id, st));
                        }
                        if let Some(p) = g.staging.remove(&job_id) {
                            let _ = fs::remove_file(p);
                        }
                    }
                    g.queue.complete(job, now);
                    return Ok((job_id, JobStatus::Succeeded));
                }
                Ok(_) => {
                    // Continue to next stage in this call (full drain of one job).
                    continue;
                }
                Err(e) => {
                    let mut g = self.inner.lock();
                    let now = g.now();
                    let stage = job.next_stage;
                    let st = g.queue.fail_stage(job, stage, e.to_string(), now);
                    return Ok((job_id, st));
                }
            }
        }
    }

    /// Drain the queue until empty or a job is only RetryWait/FailedVisible.
    /// Processes Succeeded and re-attempts RetryWait jobs once per call cycle.
    pub fn drain(&self) -> Result<Vec<(String, JobStatus)>> {
        let mut out = Vec::new();
        loop {
            match self.process_one() {
                Ok(pair) => {
                    let terminal =
                        matches!(pair.1, JobStatus::Succeeded | JobStatus::FailedVisible);
                    out.push(pair);
                    if !terminal {
                        // RetryWait requeued — keep going if something is ready.
                        continue;
                    }
                    if self.queue_len() == 0 {
                        break;
                    }
                }
                Err(ObjectError::QueueEmpty) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    fn run_job_stage(&self, job: &mut IngestJob) -> Result<bool> {
        let g = self.inner.lock();
        let now = g.now();
        let stage_path = g
            .staging
            .get(&job.id)
            .cloned()
            .ok_or_else(|| ObjectError::Invalid(format!("no staging for {}", job.id)))?;
        let mut record = g
            .in_flight
            .get(&job.id)
            .cloned()
            .ok_or_else(|| ObjectError::Invalid(format!("no in_flight for {}", job.id)))?;

        // Determine next stage from record.
        let next = StageName::pipeline()
            .iter()
            .find(|s| {
                record
                    .stage(**s)
                    .map(|r| {
                        matches!(
                            r.status,
                            crate::objects::schema::StageStatus::Pending
                                | crate::objects::schema::StageStatus::Partial
                                | crate::objects::schema::StageStatus::Failed
                        )
                    })
                    .unwrap_or(true)
            })
            .copied();

        let Some(stage) = next else {
            return Ok(true);
        };
        job.next_stage = stage;

        let hash_for_persist = record.content_hash.clone();
        let persist_dst = if stage == StageName::Persist {
            if hash_for_persist.as_str() == "blake3:pending" {
                return Err(ObjectError::StageNotReady {
                    stage: "persist".into(),
                    status: "receive incomplete".into(),
                });
            }
            Some(g.blob_path(&hash_for_persist))
        } else {
            None
        };
        drop(g);

        let outcome = pipeline::run_next_stage(
            &mut record,
            &stage_path,
            &self.processors,
            now,
            persist_dst.as_deref(),
        )?;

        let mut g = self.inner.lock();
        // Write back record.
        if let Some(slot) = g.in_flight.get_mut(&job.id) {
            *slot = record.clone();
        }
        job.content_hash = Some(record.content_hash.clone());
        job.updated_at_ms = now;
        if outcome.completed_stage == StageName::Receive {
            // Advance.
        }
        Ok(outcome.object_complete)
    }

    // ------------------------------------------------------------------
    // Read API (permissions + retention)
    // ------------------------------------------------------------------

    fn check_access(&self, rec: &ObjectRecord, reader: &Reader) -> Result<()> {
        let g = self.inner.lock();
        let now = g.now();
        let live = g.live_session.clone();
        drop(g);
        rec.permissions.check_read(reader)?;
        rec.retention.check_readable(now, live.as_deref())?;
        Ok(())
    }

    /// Look up object by content hash (metadata only).
    pub fn get_record(&self, hash: &ContentHash, reader: &Reader) -> Result<ObjectRecord> {
        let g = self.inner.lock();
        let rec = g
            .objects
            .get(hash.as_str())
            .cloned()
            .ok_or_else(|| ObjectError::NotFound(hash.as_str().into()))?;
        drop(g);
        self.check_access(&rec, reader)?;
        Ok(rec)
    }

    /// All refs pointing at a content hash.
    pub fn refs_for(&self, hash: &ContentHash) -> Vec<ObjectRef> {
        let g = self.inner.lock();
        g.refs
            .values()
            .filter(|r| r.content_hash == *hash)
            .cloned()
            .collect()
    }

    pub fn get_ref(&self, ref_id: &str) -> Result<ObjectRef> {
        let g = self.inner.lock();
        g.refs
            .get(ref_id)
            .cloned()
            .ok_or_else(|| ObjectError::RefNotFound(ref_id.into()))
    }

    /// Count of unique objects (by content hash).
    pub fn object_count(&self) -> usize {
        self.inner.lock().objects.len()
    }

    pub fn ref_count(&self) -> usize {
        self.inner.lock().refs.len()
    }

    /// Snapshot every named reference in stable id order.  This intentionally
    /// returns metadata only; callers must still use `get_record` or
    /// `compile_view_for_ref` to cross the permission boundary.
    pub fn list_refs(&self) -> Vec<ObjectRef> {
        self.inner.lock().refs.values().cloned().collect()
    }

    /// Context-compile path: derivatives only. Never raw bytes.
    pub fn compile_view(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        selection: &DerivativeSelection,
        label: Option<String>,
    ) -> Result<CompileObjectView> {
        let rec = self.get_record(hash, reader)?;
        rec.permissions.check_model_derivatives(reader)?;
        if rec.status != ObjectStatus::Ready {
            return Err(ObjectError::StageNotReady {
                stage: "finalize".into(),
                status: format!("{:?}", rec.status),
            });
        }
        Ok(CompileObjectView::from_record(
            &rec,
            &selection.kinds,
            label,
        ))
    }

    /// Compile view by ref id.
    pub fn compile_view_for_ref(
        &self,
        ref_id: &str,
        reader: &Reader,
        selection: &DerivativeSelection,
    ) -> Result<CompileObjectView> {
        let r = self.get_ref(ref_id)?;
        self.compile_view(&r.content_hash, reader, selection, r.label)
    }

    /// Raw body bytes — requires [`RawBytesCap`]. Not available on compile path.
    pub fn raw_bytes(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        _cap: &RawBytesCap,
    ) -> Result<Vec<u8>> {
        let rec = self.get_record(hash, reader)?;
        if !rec.permissions.allow_export {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_export is false".into(),
            });
        }
        match &rec.location {
            ObjectLocation::Local { path } => {
                let mut f = File::open(path)?;
                let mut out = Vec::new();
                // Still stream into vec for API simplicity; callers with large
                // files should use raw_reader. Cap via budget max_object.
                f.read_to_end(&mut out)?;
                let actual = ContentHash::of_bytes(&out);
                if actual != *hash {
                    return Err(ObjectError::ContentAddressMismatch {
                        expected: hash.as_str().into(),
                        actual: actual.as_str().into(),
                    });
                }
                Ok(out)
            }
            other => Err(ObjectError::Invalid(format!(
                "raw_bytes not available for location {other:?}"
            ))),
        }
    }

    /// Stream raw body into `writer` without requiring full RAM. Needs cap.
    pub fn raw_stream_to<W: Write>(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        _cap: &RawBytesCap,
        writer: &mut W,
    ) -> Result<usize> {
        let rec = self.get_record(hash, reader)?;
        if !rec.permissions.allow_export {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_export is false".into(),
            });
        }
        let path = match &rec.location {
            ObjectLocation::Local { path } => path.clone(),
            other => {
                return Err(ObjectError::Invalid(format!(
                    "raw stream not available for {other:?}"
                )))
            }
        };
        let mut f = File::open(path)?;
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut peak = 0usize;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            peak = peak.max(n);
            writer.write_all(&buf[..n])?;
        }
        Ok(peak)
    }

    /// Lookup job.
    pub fn job(&self, id: &str) -> Option<IngestJob> {
        self.inner.lock().queue.get(id).cloned()
    }

    /// Find object hash produced by a finished job (from in_flight merge / job).
    pub fn hash_for_job(&self, job_id: &str) -> Option<ContentHash> {
        let g = self.inner.lock();
        g.queue
            .get(job_id)
            .and_then(|j| j.content_hash.clone())
            .or_else(|| g.in_flight.get(job_id).map(|r| r.content_hash.clone()))
    }
}

fn write_bytes_chunked(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + CHUNK_SIZE).min(bytes.len());
        f.write_all(&bytes[offset..end])?;
        offset = end;
    }
    f.flush()?;
    Ok(())
}

fn copy_chunked(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut input = File::open(src)?;
    let mut output = File::create(dst)?;
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        output.write_all(&buf[..n])?;
    }
    output.flush()?;
    Ok(())
}
