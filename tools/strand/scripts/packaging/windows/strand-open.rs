// strand-open.exe — windowless double-click shim for .sa archives on Windows.
//
// Built for the GUI subsystem (`windows_subsystem = "windows"`), so launching
// it never flashes a console window — the gap the CLI's own `unpack` leaves
// when wired directly into a file association. It simply re-invokes the real
// `strand` CLI as `strand unpack <archive>` with no visible window, and on
// failure shows a single native message box.
//
// Build (no external crates, links user32 for the message box):
//   rustc -O --edition 2021 -C link-args=user32.lib \
//         --crate-name strand_open packaging/windows/strand-open.rs \
//         -o strand-open.exe
// `make-shim.ps1` wraps this and locates strand.exe next to the shim.

#![windows_subsystem = "windows"]

use std::env;
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

// CREATE_NO_WINDOW: child runs with no console, so the brief flash is gone.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

extern "system" {
    fn MessageBoxW(hwnd: *const u16, text: *const u16, caption: *const u16, utype: u32) -> i32;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn error_box(msg: &str) {
    let text = wide(msg);
    let caption = wide("STRAND");
    // MB_OK | MB_ICONERROR
    unsafe { MessageBoxW(std::ptr::null(), text.as_ptr(), caption.as_ptr(), 0x10); }
}

fn main() {
    let archive = match env::args_os().nth(1) {
        Some(a) => a,
        None => {
            error_box("Double-click a .sa archive to extract it.");
            return;
        }
    };

    // The real CLI ships next to this shim.
    let mut cli = env::current_exe().unwrap_or_default();
    cli.pop();
    let cli: PathBuf = cli.join("strand.exe");
    if !cli.exists() {
        error_box("strand.exe was not found next to strand-open.exe. Reinstall STRAND.");
        return;
    }

    match Command::new(&cli)
        .arg("unpack")
        .arg(&archive)
        .creation_flags(CREATE_NO_WINDOW)
        .status()
    {
        Ok(s) if s.success() => {}
        Ok(s) => error_box(&format!(
            "STRAND could not extract this archive (exit {}).",
            s.code().unwrap_or(-1)
        )),
        Err(e) => error_box(&format!("Failed to launch strand.exe: {e}")),
    }
}
