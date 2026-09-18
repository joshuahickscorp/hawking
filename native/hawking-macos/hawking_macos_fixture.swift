// Disposable deterministic native UI fixture for Hawking's semantic-action gate.
// It is intentionally a tiny AppKit app: one AXButton changes one AXStaticText
// from "idle" to "done".  It owns no Hawking state and is never a production
// desktop controller.

import AppKit

final class HawkingFixtureDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var state: NSTextField!
    private var button: NSButton!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let frame = NSRect(x: 0, y: 0, width: 360, height: 170)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Hawking semantic action fixture"
        window.center()

        let content = NSView(frame: frame)
        state = NSTextField(labelWithString: "idle")
        state.frame = NSRect(x: 24, y: 108, width: 312, height: 28)
        state.alignment = .center
        state.setAccessibilityIdentifier("hawking.fixture.state")
        state.setAccessibilityLabel("Fixture state")

        button = NSButton(title: "Toggle", target: self, action: #selector(toggle))
        button.frame = NSRect(x: 125, y: 42, width: 110, height: 34)
        button.bezelStyle = .rounded
        button.setAccessibilityIdentifier("hawking.fixture.toggle")
        button.setAccessibilityLabel("Toggle fixture")

        content.addSubview(state)
        content.addSubview(button)
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func toggle() {
        state.stringValue = state.stringValue == "idle" ? "done" : "idle"
        button.title = state.stringValue == "done" ? "Reset" : "Toggle"
    }
}

let application = NSApplication.shared
let delegate = HawkingFixtureDelegate()
application.delegate = delegate
// Keep the disposable fixture as a real frontmost AppKit application so the
// live lease path can observe user-focus preemption without relying on
// private or coordinate-based input APIs.
application.setActivationPolicy(.regular)
application.run()
