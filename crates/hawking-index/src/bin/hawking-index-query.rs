//! `hawking-index-query python-facts [--git-head] [--commit REV] [--repo DIR]`
//!
//! stdin: optional overlay NDJSON, one `{"path":"...","content":"..."}` per line.
//! stdout: one `hawking.index.python_facts.v1` document.
//!
//! `--git-head` / `--commit` read blobs from a git commit (default HEAD), never
//! the working tree. Overlay NDJSON still wins per path. Sparse checkouts
//! where `hcli/` is absent from disk still index those files. Untracked
//! files are invisible.

use hawking_index::python_facts::{
    default_repo, dump_python_facts_at_commit, dump_python_facts_from_overlay, read_overlay_ndjson,
    PYTHON_FACTS_SCHEMA,
};
use std::collections::HashSet;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::process;

fn usage() -> ! {
    eprintln!(
        "Usage: hawking-index-query python-facts [--git-head] [--commit REV] [--repo DIR] [--watch NAME]...\n\
         Schema: {PYTHON_FACTS_SCHEMA}\n\
         git-backed dumps read commit blobs (default HEAD), never the working tree\n\
         stdin: overlay NDJSON {{\"path\",\"content\"}} (optional)\n\
         stdout: python-facts dump"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
    }
    if args.len() < 2 {
        usage();
    }
    let cmd = args[1].as_str();
    if cmd != "python-facts" {
        eprintln!("unknown command {cmd:?}; want python-facts");
        usage();
    }

    let mut git_head = false;
    let mut commit_rev: Option<String> = None;
    let mut repo = default_repo();
    let mut watch: HashSet<String> = HashSet::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--git-head" => git_head = true,
            "--commit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--commit needs a revision");
                    process::exit(2);
                }
                commit_rev = Some(args[i].clone());
                git_head = true;
            }
            "--repo" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--repo needs a directory");
                    process::exit(2);
                }
                repo = PathBuf::from(&args[i]);
            }
            "--watch" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--watch needs a name");
                    process::exit(2);
                }
                watch.insert(args[i].clone());
            }
            other => {
                eprintln!("unknown flag {other}");
                usage();
            }
        }
        i += 1;
    }

    let overlay = if io::stdin().is_terminal() {
        Default::default()
    } else {
        match read_overlay_ndjson(io::stdin()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        }
    };

    let dump = if git_head {
        let rev = commit_rev.as_deref().unwrap_or("HEAD");
        match dump_python_facts_at_commit(&repo, rev, &overlay, &watch) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        }
    } else {
        dump_python_facts_from_overlay(&overlay, &watch)
    };

    let out = serde_json::to_string(&dump).expect("serialize python-facts");
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.write_all(b"\n");
}
