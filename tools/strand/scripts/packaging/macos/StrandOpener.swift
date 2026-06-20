// StrandOpener.swift — minimal AppKit document-opener shell for STRAND.
//
// Purpose: make `.sa` archives double-clickable. When macOS hands us one or
// more .sa documents, run the bundled `strand` CLI (Contents/MacOS/strand)
// with `unpack <file> -o <destDir>`, reveal the extraction directory in
// Finder, and quit. Headless (LSUIElement): no Dock icon, no menu bar;
// launched bare it exits silently. Alerts appear only on failure.
//
// No storyboard, no nib: a programmatic NSApplication + delegate.
// Build: xcrun swiftc -O -target arm64-apple-macos11.0 StrandOpener.swift -o StrandOpener

import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {

    /// Set as soon as macOS delivers documents, so the "launched bare"
    /// fallback in applicationDidFinishLaunching knows to stand down.
    private var receivedDocuments = false

    // MARK: - Document open path

    func application(_ application: NSApplication, open urls: [URL]) {
        receivedDocuments = true

        guard let cli = strandCLI() else {
            fail(title: "STRAND is damaged",
                 message: "The bundled 'strand' executable is missing from the app. Reinstall STRAND.app.")
            return
        }

        var revealTargets: [URL] = []
        var failures: [String] = []

        for url in urls {
            guard url.isFileURL, url.pathExtension.lowercased() == "sa" else {
                failures.append("\(url.lastPathComponent): not a .sa archive")
                continue
            }

            let dest = uniqueDestination(for: url)
            let result = run(cli: cli, arguments: ["unpack", url.path, "-o", dest.path])
            if result.status == 0 {
                revealTargets.append(dest)
            } else {
                let detail = result.stderr.isEmpty ? "exit code \(result.status)" : result.stderr
                failures.append("\(url.lastPathComponent): \(detail)")
                // Don't leave a half-written extraction dir behind on failure.
                if let contents = try? FileManager.default.contentsOfDirectory(atPath: dest.path),
                   contents.isEmpty {
                    try? FileManager.default.removeItem(at: dest)
                }
            }
        }

        if !revealTargets.isEmpty {
            NSWorkspace.shared.activateFileViewerSelecting(revealTargets)
        }

        if !failures.isEmpty {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.alertStyle = .warning
            alert.messageText = "STRAND could not extract some archives"
            alert.informativeText = failures.joined(separator: "\n")
            alert.runModal()
        }

        // Give the Finder-reveal Apple event a beat to leave the process
        // before we die.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            NSApp.terminate(nil)
        }
    }

    // MARK: - Bare launch path

    func applicationDidFinishLaunching(_ notification: Notification) {
        // application(_:open:) is delivered around didFinishLaunching when the
        // app is launched via a document; give it a moment before concluding
        // we were launched bare.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self, !self.receivedDocuments else { return }
            NSApp.terminate(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - Helpers

    /// The bundled CLI at Contents/MacOS/strand.
    private func strandCLI() -> URL? {
        if let aux = Bundle.main.url(forAuxiliaryExecutable: "strand"),
           FileManager.default.isExecutableFile(atPath: aux.path) {
            return aux
        }
        // Belt-and-braces fallback to the literal path.
        let direct = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS/strand")
        return FileManager.default.isExecutableFile(atPath: direct.path) ? direct : nil
    }

    /// destDir = archive's directory + archive filename minus extension,
    /// uniquified with " 2", " 3", ... if it already exists.
    private func uniqueDestination(for archive: URL) -> URL {
        let dir = archive.deletingLastPathComponent()
        let base = archive.deletingPathExtension().lastPathComponent
        var candidate = dir.appendingPathComponent(base, isDirectory: true)
        var n = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = dir.appendingPathComponent("\(base) \(n)", isDirectory: true)
            n += 1
        }
        return candidate
    }

    private struct RunResult {
        let status: Int32
        let stderr: String
    }

    private func run(cli: URL, arguments: [String]) -> RunResult {
        let process = Process()
        process.executableURL = cli
        process.arguments = arguments
        let errPipe = Pipe()
        process.standardOutput = Pipe()
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return RunResult(status: -1, stderr: error.localizedDescription)
        }
        process.waitUntilExit()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let errText = String(data: errData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return RunResult(status: process.terminationStatus, stderr: errText)
    }

    private func fail(title: String, message: String) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = title
        alert.informativeText = message
        alert.runModal()
        NSApp.terminate(nil)
    }
}

@main
final class Main {
    // Strong reference: NSApplication.delegate is weak.
    private static let delegate = AppDelegate()

    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        app.delegate = delegate
        app.run()
    }
}
