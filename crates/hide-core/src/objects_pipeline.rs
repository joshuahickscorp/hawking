//! Incremental processing pipeline.
//!
//! Each stage is independently resumable and recorded on the [`ObjectRecord`].
//! Streaming stages never allocate more than [`crate::objects::hash::CHUNK_SIZE`] for
//! the body working buffer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::{ContentHash, CHUNK_SIZE};
use crate::objects::kinds::ObjectKind;
use crate::objects::processors::ProcessorSet;
use crate::objects::schema::{
    Derivative, DerivativeKind, ObjectLocation, ObjectRecord, ObjectStatus, StageName,
    StageRecord, StageStatus,
};

/// Outcome of running stages for one object.
#[derive(Debug, Clone)]
pub struct StageOutcome {
    pub peak_buffer_bytes: usize,
    pub completed_stage: StageName,
    pub object_complete: bool,
}

fn ensure_stage(record: &mut ObjectRecord, stage: StageName, now: u64) {
    if record.stage(stage).is_none() {
        record.stages.push(StageRecord::pending(stage, now));
    }
}

fn mark_running(record: &mut ObjectRecord, stage: StageName, now: u64) {
    ensure_stage(record, stage, now);
    if let Some(st) = record.stage_mut(stage) {
        st.status = StageStatus::Running;
        st.attempts += 1;
        st.updated_at_ms = now;
    }
}

fn mark_complete(
    record: &mut ObjectRecord,
    stage: StageName,
    now: u64,
    bytes_processed: u64,
    bytes_total: Option<u64>,
    peak: usize,
) {
    if let Some(st) = record.stage_mut(stage) {
        st.bytes_processed = bytes_processed;
        st.bytes_total = bytes_total;
        st.peak_buffer_bytes = peak.max(st.peak_buffer_bytes);
        st.status = StageStatus::Complete;
        st.updated_at_ms = now;
        st.last_error = None;
    }
    record.updated_at_ms = now;
}

fn mark_skipped(record: &mut ObjectRecord, stage: StageName, now: u64, peak: usize) {
    if let Some(st) = record.stage_mut(stage) {
        st.peak_buffer_bytes = peak.max(st.peak_buffer_bytes);
        st.status = StageStatus::Skipped;
        st.updated_at_ms = now;
    }
    record.updated_at_ms = now;
}

fn mark_failed(record: &mut ObjectRecord, stage: StageName, now: u64, err: impl Into<String>) {
    if let Some(st) = record.stage_mut(stage) {
        st.status = StageStatus::Failed;
        st.last_error = Some(err.into());
        st.updated_at_ms = now;
    }
    record.updated_at_ms = now;
}

/// Stream-copy `src` → `dst`, hashing, with optional resume from `bytes_done`.
pub fn stream_persist(
    src: &Path,
    dst: &Path,
    bytes_done: u64,
) -> Result<(Option<ContentHash>, u64, usize)> {
    let mut input = File::open(src)?;
    let src_meta = input.metadata()?;
    let total = src_meta.len();

    if bytes_done > total {
        return Err(ObjectError::Invalid(format!(
            "resume offset {bytes_done} past end {total}"
        )));
    }

    input.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; CHUNK_SIZE];
    let mut peak = 0usize;
    let mut read_total = 0u64;

    let mut output = if bytes_done == 0 {
        File::create(dst)?
    } else {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            // Resume path: keep the bytes already fetched, then seek past them.
            .truncate(false)
            .open(dst)?;
        f.set_len(bytes_done)?;
        f.seek(SeekFrom::Start(bytes_done))?;
        f
    };

    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        peak = peak.max(n);
        hasher.update(&buf[..n]);
        let chunk_start = read_total;
        let chunk_end = read_total + n as u64;
        if chunk_end > bytes_done {
            let skip = bytes_done.saturating_sub(chunk_start) as usize;
            output.write_all(&buf[skip..n])?;
        }
        read_total = chunk_end;
    }
    output.flush()?;

    let hex = hasher.finalize().to_hex();
    let hash = ContentHash(format!("blake3:{hex}"));
    Ok((Some(hash), read_total, peak))
}

/// Hash a file without loading it whole.
pub fn hash_file(path: &Path) -> Result<(ContentHash, u64, usize)> {
    let f = File::open(path)?;
    let (h, size, peak) = ContentHash::of_reader(f)?;
    Ok((h, size, peak))
}

/// Run the next pending stage for `record`, using body at `body_path`.
pub fn run_next_stage(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now_ms: u64,
    persist_dst: Option<&Path>,
) -> Result<StageOutcome> {
    let stage = next_incomplete_stage(record).ok_or_else(|| {
        ObjectError::Invalid("no incomplete stage".into())
    })?;

    match stage {
        StageName::Receive => run_receive(record, body_path, now_ms),
        StageName::Persist => {
            let dst = persist_dst.ok_or_else(|| {
                ObjectError::Invalid("persist_dst required for Persist stage".into())
            })?;
            run_persist(record, body_path, dst, now_ms)
        }
        StageName::Classify => run_classify(record, now_ms),
        StageName::ExtractText => run_extract_text(record, body_path, processors, now_ms),
        StageName::OcrOrTranscript => run_ocr_or_transcript(record, body_path, processors, now_ms),
        StageName::Thumbnail => run_thumbnail(record, body_path, processors, now_ms),
        StageName::Finalize => run_finalize(record, now_ms),
    }
}

fn next_incomplete_stage(record: &ObjectRecord) -> Option<StageName> {
    for s in StageName::pipeline() {
        match record.stage(*s) {
            None => return Some(*s),
            Some(r)
                if matches!(
                    r.status,
                    StageStatus::Pending | StageStatus::Partial | StageStatus::Failed
                ) =>
            {
                return Some(*s);
            }
            Some(r) if matches!(r.status, StageStatus::Complete | StageStatus::Skipped) => {
                continue;
            }
            Some(_) => return Some(*s),
        }
    }
    None
}

fn run_receive(record: &mut ObjectRecord, body_path: &Path, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Receive, now);

    let (hash, size, peak) = hash_file(body_path)?;
    let pending = record.content_hash.as_str().is_empty()
        || record.content_hash.as_str() == "blake3:pending";
    if pending {
        record.content_hash = hash.clone();
    } else if record.content_hash != hash {
        mark_failed(record, StageName::Receive, now, "hash mismatch on receive");
        return Err(ObjectError::ContentAddressMismatch {
            expected: record.content_hash.as_str().into(),
            actual: hash.as_str().into(),
        });
    }
    record.size_bytes = size;
    record.status = ObjectStatus::Processing;
    mark_complete(record, StageName::Receive, now, size, Some(size), peak);

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Receive,
        object_complete: false,
    })
}

fn run_persist(
    record: &mut ObjectRecord,
    body_path: &Path,
    dst: &Path,
    now: u64,
) -> Result<StageOutcome> {
    let already = record
        .stage(StageName::Persist)
        .map(|s| s.bytes_processed)
        .unwrap_or(0);
    mark_running(record, StageName::Persist, now);

    if dst.exists() {
        let (h, size, peak) = hash_file(dst)?;
        if h == record.content_hash {
            record.location = ObjectLocation::Local {
                path: dst.display().to_string(),
            };
            mark_complete(record, StageName::Persist, now, size, Some(size), peak);
            return Ok(StageOutcome {
                peak_buffer_bytes: peak,
                completed_stage: StageName::Persist,
                object_complete: false,
            });
        }
    }

    let expected = record.content_hash.clone();
    let (hash, total, peak) = stream_persist(body_path, dst, already)?;
    if let Some(h) = hash {
        if h != expected {
            mark_failed(record, StageName::Persist, now, "persist hash mismatch");
            return Err(ObjectError::ContentAddressMismatch {
                expected: expected.as_str().into(),
                actual: h.as_str().into(),
            });
        }
    }

    record.location = ObjectLocation::Local {
        path: dst.display().to_string(),
    };
    mark_complete(record, StageName::Persist, now, total, Some(total), peak);

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Persist,
        object_complete: false,
    })
}

fn run_classify(record: &mut ObjectRecord, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Classify, now);
    record.kind = ObjectKind::from_mime(&record.mime);
    mark_complete(record, StageName::Classify, now, 0, None, 0);
    Ok(StageOutcome {
        peak_buffer_bytes: 0,
        completed_stage: StageName::Classify,
        object_complete: false,
    })
}

/// Read up to `max_in_memory` bytes; still scans the full file for peak tracking.
fn read_body_bounded(path: &Path, max_in_memory: usize) -> Result<(Vec<u8>, usize, u64)> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; CHUNK_SIZE];
    let mut peak = 0usize;
    let mut total = 0u64;
    loop {
        let n = f.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        peak = peak.max(n);
        total += n as u64;
        if buf.len() < max_in_memory {
            let take = (max_in_memory - buf.len()).min(n);
            buf.extend_from_slice(&chunk[..take]);
        }
    }
    Ok((buf, peak, total))
}

const PROCESSOR_IN_MEMORY_CAP: usize = 4 * 1024 * 1024;

fn run_extract_text(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::ExtractText, now);
    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let mime = record.mime.clone();

    if let Some(out) = processors.text.extract(&mime, &body) {
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: out.text.len() as u64,
            inline_text: Some(out.text),
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(record, StageName::ExtractText, now, total, Some(total), peak);
    } else {
        mark_skipped(record, StageName::ExtractText, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::ExtractText,
        object_complete: false,
    })
}

fn run_ocr_or_transcript(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::OcrOrTranscript, now);

    if !record.kind.requires_ocr_or_transcript() {
        mark_skipped(record, StageName::OcrOrTranscript, now, 0);
        return Ok(StageOutcome {
            peak_buffer_bytes: 0,
            completed_stage: StageName::OcrOrTranscript,
            object_complete: false,
        });
    }

    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let mime = record.mime.clone();

    let out = processors
        .ocr
        .ocr(&mime, &body)
        .or_else(|| processors.asr.transcribe(&mime, &body));

    if let Some(out) = out {
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: out.text.len() as u64,
            inline_text: Some(out.text),
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(
            record,
            StageName::OcrOrTranscript,
            now,
            total,
            Some(total),
            peak,
        );
    } else {
        mark_skipped(record, StageName::OcrOrTranscript, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::OcrOrTranscript,
        object_complete: false,
    })
}

fn run_thumbnail(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::Thumbnail, now);
    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let kind = record.kind;
    let mime = record.mime.clone();

    if let Some(out) = processors.thumb.thumbnail(kind, &mime, &body) {
        let dhash = ContentHash::of_bytes(&out.bytes);
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: Some(dhash),
            mime: out.mime,
            size_bytes: out.bytes.len() as u64,
            inline_text: None,
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(record, StageName::Thumbnail, now, total, Some(total), peak);
    } else {
        mark_skipped(record, StageName::Thumbnail, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Thumbnail,
        object_complete: false,
    })
}

fn run_finalize(record: &mut ObjectRecord, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Finalize, now);

    if record.derivative(DerivativeKind::Summary).is_none() {
        let summary = format!(
            "object kind={} mime={} size_bytes={} hash={}",
            record.kind.as_str(),
            record.mime,
            record.size_bytes,
            record.content_hash
        );
        record.derivatives.push(Derivative {
            kind: DerivativeKind::Summary,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: summary.len() as u64,
            inline_text: Some(summary),
            produced_by: "finalize_summary".into(),
            produced_at_ms: now,
        });
    }

    record.status = ObjectStatus::Ready;
    mark_complete(record, StageName::Finalize, now, 0, None, 0);

    Ok(StageOutcome {
        peak_buffer_bytes: 0,
        completed_stage: StageName::Finalize,
        object_complete: true,
    })
}

/// Seed a new record with all stages Pending and a placeholder hash.
pub fn new_processing_record(
    mime: String,
    source: crate::objects::schema::ObjectSource,
    permissions: crate::objects::permissions::ObjectPermissions,
    retention: crate::objects::retention::RetentionPolicy,
    now: u64,
) -> ObjectRecord {
    let stages = StageName::pipeline()
        .iter()
        .map(|s| StageRecord::pending(*s, now))
        .collect();
    ObjectRecord {
        content_hash: ContentHash("blake3:pending".into()),
        mime,
        kind: ObjectKind::Other,
        size_bytes: 0,
        source,
        location: ObjectLocation::Pending,
        status: ObjectStatus::Queued,
        stages,
        derivatives: Vec::new(),
        permissions,
        retention,
        created_at_ms: now,
        updated_at_ms: now,
    }
}
