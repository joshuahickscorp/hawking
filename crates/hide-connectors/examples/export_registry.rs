//! Write HIDE_YOU_CONNECTOR_REGISTRY.json to the path given as argv[1], or stdout.
use std::env;
use std::io::Write;
use std::path::PathBuf;

use hide_connectors::ConnectorRegistry;

fn main() {
    let reg = ConnectorRegistry::builtin();
    reg.validate_all().expect("ABI validation");
    let doc = reg.export_document();
    let text = serde_json::to_string_pretty(&doc).expect("serialize") + "\n";
    match env::args().nth(1) {
        Some(path) => {
            let p = PathBuf::from(path);
            std::fs::write(&p, &text).expect("write");
            eprintln!("wrote {}", p.display());
        }
        None => {
            let mut out = std::io::stdout().lock();
            out.write_all(text.as_bytes()).expect("stdout");
        }
    }
}
