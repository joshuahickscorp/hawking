//! Processing interfaces and honestly-named fakes.
//!
//! Real OCR / ASR / vision models are out of scope. The pipeline is real; the
//! engines are labelled `Fake*` so nothing pretends to be a model.

use crate::objects::kinds::ObjectKind;
use crate::objects::schema::DerivativeKind;

/// Result of a text-oriented derivative stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextDerivativeOut {
    pub kind: DerivativeKind,
    pub text: String,
    pub produced_by: String,
}

/// Result of a binary derivative (thumbnail / proxy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDerivativeOut {
    pub kind: DerivativeKind,
    pub bytes: Vec<u8>,
    pub mime: String,
    pub produced_by: String,
}

/// Extract plain text when the body is already textual.
pub trait TextExtractor: Send + Sync {
    fn name(&self) -> &'static str;
    fn extract(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// OCR for images / scanned PDFs. Real models are not loaded here.
pub trait OcrEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn ocr(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// Speech-to-text for audio / video. Real models are not loaded here.
pub trait AsrEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn transcribe(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// Thumbnail / proxy generation.
pub trait Thumbnailer: Send + Sync {
    fn name(&self) -> &'static str;
    fn thumbnail(&self, kind: ObjectKind, mime: &str, body: &[u8]) -> Option<BinaryDerivativeOut>;
}

// ---------------------------------------------------------------------------
// Real lightweight extractors (no model)
// ---------------------------------------------------------------------------

/// Deterministic UTF-8 text extract for `text/*` and a few code MIME types.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utf8TextExtractor;

impl TextExtractor for Utf8TextExtractor {
    fn name(&self) -> &'static str {
        "utf8_text_extract"
    }

    fn extract(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        let textual = m.starts_with("text/")
            || m == "application/json"
            || m == "application/xml"
            || m.contains("javascript")
            || m.contains("typescript")
            || m.contains("python")
            || m.contains("rust");
        if !textual {
            return None;
        }
        let text = String::from_utf8_lossy(body).into_owned();
        Some(TextDerivativeOut {
            kind: DerivativeKind::TextExtract,
            text,
            produced_by: self.name().into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Honestly-named fakes
// ---------------------------------------------------------------------------

/// Fake OCR. Labels itself `FakeOcrEngine`. Not a real vision model.
///
/// Produces a deterministic placeholder string from a blake3 prefix of the body
/// so tests can assert non-empty OCR without loading models.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeOcrEngine;

impl OcrEngine for FakeOcrEngine {
    fn name(&self) -> &'static str {
        "FakeOcrEngine"
    }

    fn ocr(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        if !(m.starts_with("image/") || m == "application/pdf") {
            return None;
        }
        let digest = blake3::hash(body).to_hex();
        let text = format!(
            "[FakeOcrEngine] placeholder OCR for {mime} (blake3_prefix={})",
            &digest[..12]
        );
        Some(TextDerivativeOut {
            kind: DerivativeKind::Ocr,
            text,
            produced_by: self.name().into(),
        })
    }
}

/// Fake ASR. Labels itself `FakeAsrEngine`. Not a real speech model.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeAsrEngine;

impl AsrEngine for FakeAsrEngine {
    fn name(&self) -> &'static str {
        "FakeAsrEngine"
    }

    fn transcribe(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        if !(m.starts_with("audio/") || m.starts_with("video/")) {
            return None;
        }
        let digest = blake3::hash(body).to_hex();
        let text = format!(
            "[FakeAsrEngine] placeholder transcript for {mime} (blake3_prefix={})",
            &digest[..12]
        );
        Some(TextDerivativeOut {
            kind: DerivativeKind::Transcript,
            text,
            produced_by: self.name().into(),
        })
    }
}

/// Fake thumbnailer. Labels itself `FakeThumbnailer`. Emits a tiny deterministic
/// "proxy" blob, not a real image codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeThumbnailer;

impl Thumbnailer for FakeThumbnailer {
    fn name(&self) -> &'static str {
        "FakeThumbnailer"
    }

    fn thumbnail(&self, kind: ObjectKind, mime: &str, body: &[u8]) -> Option<BinaryDerivativeOut> {
        // Only for kinds that usually get a visual proxy.
        if !matches!(
            kind,
            ObjectKind::Image
                | ObjectKind::Pdf
                | ObjectKind::Video
                | ObjectKind::DesignFile
                | ObjectKind::Slides
        ) {
            return None;
        }
        let digest = blake3::hash(body);
        let mut bytes = b"FAKE_THUMB:".to_vec();
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(mime.as_bytes());
        Some(BinaryDerivativeOut {
            kind: DerivativeKind::Thumbnail,
            bytes,
            mime: "application/x-fake-thumbnail".into(),
            produced_by: self.name().into(),
        })
    }
}

/// Bundle of processors used by the pipeline.
pub struct ProcessorSet {
    pub text: Box<dyn TextExtractor>,
    pub ocr: Box<dyn OcrEngine>,
    pub asr: Box<dyn AsrEngine>,
    pub thumb: Box<dyn Thumbnailer>,
}

impl Default for ProcessorSet {
    fn default() -> Self {
        Self {
            text: Box::new(Utf8TextExtractor),
            ocr: Box::new(FakeOcrEngine),
            asr: Box::new(FakeAsrEngine),
            thumb: Box::new(FakeThumbnailer),
        }
    }
}

impl ProcessorSet {
    pub fn fake_defaults() -> Self {
        Self::default()
    }
}
