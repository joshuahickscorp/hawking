//! Object kinds and MIME helpers.

use serde::{Deserialize, Serialize};

/// First-class object kinds for the YOU surface (and shared by CHAT / IDE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    Image,
    Pdf,
    Document,
    Spreadsheet,
    Slides,
    Audio,
    Video,
    Archive,
    Code,
    WebCapture,
    Asset3d,
    DesignFile,
    EmailAttachment,
    ConnectorObject,
    /// Fallback when MIME is unknown or unmapped.
    Other,
}

impl ObjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Pdf => "pdf",
            Self::Document => "document",
            Self::Spreadsheet => "spreadsheet",
            Self::Slides => "slides",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Archive => "archive",
            Self::Code => "code",
            Self::WebCapture => "web_capture",
            Self::Asset3d => "asset_3d",
            Self::DesignFile => "design_file",
            Self::EmailAttachment => "email_attachment",
            Self::ConnectorObject => "connector_object",
            Self::Other => "other",
        }
    }

    /// All first-class kinds the schema admits.
    pub fn all_first_class() -> &'static [ObjectKind] {
        &[
            Self::Image,
            Self::Pdf,
            Self::Document,
            Self::Spreadsheet,
            Self::Slides,
            Self::Audio,
            Self::Video,
            Self::Archive,
            Self::Code,
            Self::WebCapture,
            Self::Asset3d,
            Self::DesignFile,
            Self::EmailAttachment,
            Self::ConnectorObject,
        ]
    }

    /// Infer kind from a MIME type string.
    pub fn from_mime(mime: &str) -> Self {
        let m = mime.to_ascii_lowercase();
        if m.starts_with("image/") {
            return Self::Image;
        }
        if m == "application/pdf" {
            return Self::Pdf;
        }
        if m.starts_with("audio/") {
            return Self::Audio;
        }
        if m.starts_with("video/") {
            return Self::Video;
        }
        if matches!(
            m.as_str(),
            "application/zip"
                | "application/x-tar"
                | "application/gzip"
                | "application/x-7z-compressed"
                | "application/x-rar-compressed"
        ) {
            return Self::Archive;
        }
        if m.contains("spreadsheet")
            || m.contains("excel")
            || m == "text/csv"
            || m.ends_with("sheet")
        {
            return Self::Spreadsheet;
        }
        if m.contains("presentation") || m.contains("powerpoint") {
            return Self::Slides;
        }
        if m.contains("javascript")
            || m.contains("typescript")
            || m.contains("python")
            || m.contains("rust")
            || m == "text/x-rust"
            || m == "text/x-python"
        {
            return Self::Code;
        }
        if m == "text/html" {
            return Self::WebCapture;
        }
        if m.starts_with("text/")
            || m == "application/json"
            || m == "application/xml"
            || m.contains("wordprocessing")
            || m.contains("msword")
        {
            return Self::Document;
        }
        if m.starts_with("model/") || m.contains("gltf") || m.contains("mesh") {
            return Self::Asset3d;
        }
        if m.contains("photoshop")
            || m.contains("illustrator")
            || m.contains("figma")
            || m == "application/postscript"
            || m == "image/vnd.adobe.photoshop"
        {
            return Self::DesignFile;
        }
        if m == "message/rfc822" || m.contains("email") {
            return Self::EmailAttachment;
        }
        Self::Other
    }

    /// Whether OCR or ASR is typically required before usable text is available.
    pub fn requires_ocr_or_transcript(self) -> bool {
        matches!(self, Self::Image | Self::Pdf | Self::Audio | Self::Video)
    }
}

/// Best-effort MIME guess from filename extension (not authoritative).
pub fn mime_from_filename(name: &str) -> String {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "json" => "application/json",
        "rs" => "text/x-rust",
        "py" => "text/x-python",
        "ts" => "text/typescript",
        "js" => "text/javascript",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "zip" => "application/zip",
        "tar" => "application/x-tar",
        "gz" => "application/gzip",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "glb" | "gltf" => "model/gltf-binary",
        "eml" => "message/rfc822",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mime_maps_to_kinds() {
        assert_eq!(ObjectKind::from_mime("image/png"), ObjectKind::Image);
        assert_eq!(ObjectKind::from_mime("application/pdf"), ObjectKind::Pdf);
        assert_eq!(ObjectKind::from_mime("video/mp4"), ObjectKind::Video);
        assert_eq!(ObjectKind::from_mime("audio/mpeg"), ObjectKind::Audio);
        assert_eq!(ObjectKind::from_mime("text/x-rust"), ObjectKind::Code);
    }
}
