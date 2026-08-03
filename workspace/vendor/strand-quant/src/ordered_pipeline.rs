//! Bounded, deterministic three-stage pipeline for offline encoding.
//!
//! A producer owns source-read plus deterministic preprocessing, the caller owns
//! encoding, and a sink thread owns ordered output writes. Both hand-offs use
//! bounded synchronous channels. Sequence numbers are checked at every boundary;
//! no stage is allowed to reorder records silently.
//!
//! Caller-owned input descriptors and encoder scratch are outside the stage-byte
//! budgets. Production callers should keep descriptors lightweight, load source
//! bytes inside `read_prepare`, and cap encoder scratch separately.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineConfig {
    /// Maximum queued records at either stage boundary.
    pub depth: usize,
    /// Aggregate bound for prepared records: queued records plus the producer's
    /// active record and the encoder's active record.
    pub prepared_resident_budget_bytes: usize,
    /// Aggregate bound for encoded records: queued records plus the encoder's
    /// active record and the writer's active record.
    pub encoded_resident_budget_bytes: usize,
}

impl PipelineConfig {
    pub fn new(
        depth: usize,
        prepared_resident_budget_bytes: usize,
        encoded_resident_budget_bytes: usize,
    ) -> Result<Self, PipelineError> {
        if depth == 0 {
            return Err(PipelineError::config(
                "pipeline depth must be greater than zero",
            ));
        }
        if prepared_resident_budget_bytes == 0 || encoded_resident_budget_bytes == 0 {
            return Err(PipelineError::config(
                "pipeline resident byte budgets must be greater than zero",
            ));
        }
        Ok(Self {
            depth,
            prepared_resident_budget_bytes,
            encoded_resident_budget_bytes,
        })
    }

    #[inline]
    fn resident_slots(self) -> usize {
        self.depth.saturating_add(2)
    }

    #[inline]
    fn prepared_item_limit(self) -> usize {
        self.prepared_resident_budget_bytes / self.resident_slots()
    }

    #[inline]
    fn encoded_item_limit(self) -> usize {
        self.encoded_resident_budget_bytes / self.resident_slots()
    }
}

#[derive(Debug)]
pub struct Accounted<T> {
    pub value: T,
    /// Conservative resident bytes held by this record at its stage boundary.
    pub resident_bytes: usize,
}

impl<T> Accounted<T> {
    pub fn new(value: T, resident_bytes: usize) -> Self {
        Self {
            value,
            resident_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    Config,
    ReadPrepare,
    Encode,
    Write,
    Coordination,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineError {
    pub stage: PipelineStage,
    pub index: Option<usize>,
    pub message: String,
}

impl PipelineError {
    fn config(message: impl Into<String>) -> Self {
        Self {
            stage: PipelineStage::Config,
            index: None,
            message: message.into(),
        }
    }

    fn at(stage: PipelineStage, index: usize, message: impl Into<String>) -> Self {
        Self {
            stage,
            index: Some(index),
            message: message.into(),
        }
    }

    fn coordination(message: impl Into<String>) -> Self {
        Self {
            stage: PipelineStage::Coordination,
            index: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(index) = self.index {
            write!(
                f,
                "{:?} stage at record {index}: {}",
                self.stage, self.message
            )
        } else {
            write!(f, "{:?} stage: {}", self.stage, self.message)
        }
    }
}

impl std::error::Error for PipelineError {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub records_read_prepared: usize,
    pub records_encoded: usize,
    pub records_written: usize,
    /// Includes queued records and records actively owned by adjacent stages.
    pub max_prepared_resident_records: usize,
    /// Includes queued records and records actively owned by adjacent stages.
    pub max_encoded_resident_records: usize,
}

struct Envelope<T> {
    index: usize,
    accounted: Accounted<T>,
}

fn update_max(maximum: &AtomicUsize, value: usize) {
    let mut seen = maximum.load(Ordering::Relaxed);
    while value > seen {
        match maximum.compare_exchange_weak(seen, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => seen = actual,
        }
    }
}

/// Run an ordered read/preprocess → encode → write pipeline.
///
/// `read_prepare` and `write` execute on dedicated scoped threads. `encode`
/// executes on the calling thread and may itself use the block-parallel encoder.
/// A record may advance only in monotonically increasing source order. Any stage
/// error, order mismatch, panic, channel closure, or resident-budget violation
/// fails the complete pipeline; no fallback output is synthesized.
pub fn run_ordered_pipeline<I, P, O, ReadPrepare, Encode, Write, RE, EE, WE>(
    inputs: Vec<I>,
    config: PipelineConfig,
    mut read_prepare: ReadPrepare,
    mut encode: Encode,
    mut write: Write,
) -> Result<PipelineStats, PipelineError>
where
    I: Send,
    P: Send,
    O: Send,
    ReadPrepare: FnMut(usize, I) -> Result<Accounted<P>, RE> + Send,
    Encode: FnMut(usize, P) -> Result<Accounted<O>, EE>,
    Write: FnMut(usize, O) -> Result<(), WE> + Send,
    RE: fmt::Display,
    EE: fmt::Display,
    WE: fmt::Display,
{
    let resident_slots = config.resident_slots();
    let prepared_item_limit = config.prepared_item_limit();
    let encoded_item_limit = config.encoded_item_limit();
    if prepared_item_limit == 0 || encoded_item_limit == 0 {
        return Err(PipelineError::config(format!(
            "resident budgets must allow at least one byte in each of {resident_slots} slots"
        )));
    }

    let total = inputs.len();
    let prepared_resident = Arc::new(AtomicUsize::new(0));
    let encoded_resident = Arc::new(AtomicUsize::new(0));
    let max_prepared = Arc::new(AtomicUsize::new(0));
    let max_encoded = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        let (prepared_tx, prepared_rx) = mpsc::sync_channel(config.depth);
        let producer_resident = Arc::clone(&prepared_resident);
        let producer_max = Arc::clone(&max_prepared);
        let producer = scope.spawn(move || -> Result<usize, PipelineError> {
            let mut count = 0usize;
            for (index, input) in inputs.into_iter().enumerate() {
                let accounted = read_prepare(index, input).map_err(|error| {
                    PipelineError::at(PipelineStage::ReadPrepare, index, error.to_string())
                })?;
                if accounted.resident_bytes > prepared_item_limit {
                    return Err(PipelineError::at(
                        PipelineStage::ReadPrepare,
                        index,
                        format!(
                            "prepared record uses {} bytes, per-slot limit is {} (aggregate budget {}, slots {})",
                            accounted.resident_bytes,
                            prepared_item_limit,
                            config.prepared_resident_budget_bytes,
                            resident_slots,
                        ),
                    ));
                }
                let resident_now = producer_resident.fetch_add(1, Ordering::Relaxed) + 1;
                update_max(&producer_max, resident_now);
                if prepared_tx.send(Envelope { index, accounted }).is_err() {
                    producer_resident.fetch_sub(1, Ordering::Relaxed);
                    return Err(PipelineError::at(
                        PipelineStage::Coordination,
                        index,
                        "encoder closed the prepared-record channel",
                    ));
                }
                count += 1;
            }
            Ok(count)
        });

        let (encoded_tx, encoded_rx) = mpsc::sync_channel::<Envelope<O>>(config.depth);
        let writer_resident = Arc::clone(&encoded_resident);
        let writer = scope.spawn(move || -> Result<usize, PipelineError> {
            let mut expected = 0usize;
            while let Ok(envelope) = encoded_rx.recv() {
                if envelope.index != expected {
                    writer_resident.fetch_sub(1, Ordering::Relaxed);
                    return Err(PipelineError::at(
                        PipelineStage::Write,
                        envelope.index,
                        format!("out-of-order record: expected {expected}"),
                    ));
                }
                let result = write(envelope.index, envelope.accounted.value).map_err(|error| {
                    PipelineError::at(PipelineStage::Write, envelope.index, error.to_string())
                });
                writer_resident.fetch_sub(1, Ordering::Relaxed);
                result?;
                expected += 1;
            }
            Ok(expected)
        });

        let mut encoded_count = 0usize;
        let mut primary_error = None;
        for expected in 0..total {
            let envelope = match prepared_rx.recv() {
                Ok(envelope) => envelope,
                Err(_) => {
                    primary_error = Some(PipelineError::at(
                        PipelineStage::Coordination,
                        expected,
                        "producer closed before all source records arrived",
                    ));
                    break;
                }
            };
            if envelope.index != expected {
                prepared_resident.fetch_sub(1, Ordering::Relaxed);
                primary_error = Some(PipelineError::at(
                    PipelineStage::Encode,
                    envelope.index,
                    format!("out-of-order prepared record: expected {expected}"),
                ));
                break;
            }
            let encoded = encode(envelope.index, envelope.accounted.value).map_err(|error| {
                PipelineError::at(PipelineStage::Encode, envelope.index, error.to_string())
            });
            prepared_resident.fetch_sub(1, Ordering::Relaxed);
            let accounted = match encoded {
                Ok(value) => value,
                Err(error) => {
                    primary_error = Some(error);
                    break;
                }
            };
            if accounted.resident_bytes > encoded_item_limit {
                primary_error = Some(PipelineError::at(
                    PipelineStage::Encode,
                    envelope.index,
                    format!(
                        "encoded record uses {} bytes, per-slot limit is {} (aggregate budget {}, slots {})",
                        accounted.resident_bytes,
                        encoded_item_limit,
                        config.encoded_resident_budget_bytes,
                        resident_slots,
                    ),
                ));
                break;
            }
            let resident_now = encoded_resident.fetch_add(1, Ordering::Relaxed) + 1;
            update_max(&max_encoded, resident_now);
            if encoded_tx
                .send(Envelope {
                    index: envelope.index,
                    accounted,
                })
                .is_err()
            {
                encoded_resident.fetch_sub(1, Ordering::Relaxed);
                primary_error = Some(PipelineError::at(
                    PipelineStage::Coordination,
                    envelope.index,
                    "writer closed the encoded-record channel",
                ));
                break;
            }
            encoded_count += 1;
        }

        drop(prepared_rx);
        drop(encoded_tx);

        let producer_result = producer
            .join()
            .map_err(|_| PipelineError::coordination("read/preprocess worker panicked"));
        let writer_result = writer
            .join()
            .map_err(|_| PipelineError::coordination("writer worker panicked"));

        if let Some(error) = primary_error {
            match &writer_result {
                Err(join_error) => return Err(join_error.clone()),
                Ok(Err(writer_error)) => return Err(writer_error.clone()),
                Ok(Ok(_)) => {}
            }
            match &producer_result {
                Err(join_error) => return Err(join_error.clone()),
                Ok(Err(producer_error)) => return Err(producer_error.clone()),
                Ok(Ok(_)) => {}
            }
            return Err(error);
        }
        let read_count = producer_result??;
        let written_count = writer_result??;
        if read_count != total || encoded_count != total || written_count != total {
            return Err(PipelineError::coordination(format!(
                "incomplete pipeline: expected {total}, read {read_count}, encoded {encoded_count}, written {written_count}"
            )));
        }
        Ok(PipelineStats {
            records_read_prepared: read_count,
            records_encoded: encoded_count,
            records_written: written_count,
            max_prepared_resident_records: max_prepared.load(Ordering::Relaxed),
            max_encoded_resident_records: max_encoded.load(Ordering::Relaxed),
        })
    })
}
