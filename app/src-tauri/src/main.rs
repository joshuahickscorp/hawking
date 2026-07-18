// HIDE desktop shell (Tauri v2). Two jobs: (1) apply native macOS vibrancy behind the transparent
// window so the OS desktop and the colorful apps behind HIDE show through the glass (the real Liquid
// Glass the web build cannot achieve); (2) supervise the local `hide-serve` engine for the app's
// lifetime so the front end has a backend to talk to, and stop it cleanly on quit.
//
// Build-ready scaffold: needs the tauri crates fetched (network) and `pnpm add -D @tauri-apps/cli`.
// On macOS the webview is WKWebView (WebKit-class), so the glass renders through the frost path.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::{Manager, RunEvent, WindowEvent};

#[cfg(target_os = "macos")]
use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial, NSVisualEffectState};

/// Holds the supervised `hide-serve` child so we can stop it on quit.
struct Engine(Mutex<Option<std::process::Child>>);

fn start(bin: impl AsRef<std::ffi::OsStr>) -> Option<std::process::Child> {
    // hide-serve accepts `--port N` (and env HIDE_SERVE_ADDR); it bails on `--addr`, which would
    // exit the sidecar at boot. Bind loopback via HIDE_SERVE_ADDR so the host is explicit.
    std::process::Command::new(bin)
        .arg("--port")
        .arg("8744")
        .env("HIDE_SERVE_ADDR", "127.0.0.1:8744")
        .spawn()
        .ok()
}

fn spawn_engine() -> Option<std::process::Child> {
    // Resolution order:
    //   1. HIDE_SERVE_BIN env override (dev / custom),
    //   2. the bundled sidecar next to the app binary (externalBin -> Contents/MacOS/hide-serve),
    //   3. `hide-serve` on PATH (dev fallback).
    if let Ok(bin) = std::env::var("HIDE_SERVE_BIN") {
        if let Some(child) = start(&bin) {
            return Some(child);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join("hide-serve");
            if bundled.exists() {
                if let Some(child) = start(&bundled) {
                    return Some(child);
                }
            }
        }
    }
    start("hide-serve")
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Engine(Mutex::new(None)))
        .setup(|app| {
            // Vibrancy behind the transparent window (macOS only). The web glass (rim + grain) layers
            // on top of this, so the result is a true pane of glass over whatever is behind the window.
            #[cfg(target_os = "macos")]
            if let Some(win) = app.get_webview_window("main") {
                let _ = apply_vibrancy(
                    &win,
                    NSVisualEffectMaterial::HudWindow,
                    Some(NSVisualEffectState::Active),
                    None,
                );
            }
            // Bring up the local engine.
            let engine = app.state::<Engine>();
            *engine.0.lock().unwrap() = spawn_engine();
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                // Stop the engine when the last window closes.
                if let Some(engine) = window.app_handle().try_state::<Engine>() {
                    if let Some(mut child) = engine.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error building HIDE")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                if let Some(engine) = app_handle.try_state::<Engine>() {
                    if let Some(mut child) = engine.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
